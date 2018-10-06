"""Wavefront .obj file loader.

Supports basic material and texture properties.
Vertex colors are not supported.
"""

import re
import os
import numpy as np
import logging
import math
from collections import defaultdict, namedtuple

logger = logging.getLogger(__name__)


def checked_dtype_narrowing(array, target_dtype, dtype_name=""):
    """Cast a numpy array to a smaller dtype.  Asserts that underflow and
        overflow will not occur

    Parameters
    ----------
    array : Any object that can be passed to np.array
        The array to be cast.  Will be more efficient if it is already an array.
    target_dtype : dtype
        What the array should be case to, i.e., numpy.uint8
    dtype_name : string
        Optional name for the dtype for the error message

    Returns
    -------
    new_array : np.array of dtype target_dtype
    """
    array = np.asarray(array)

    if array.size > 0:
        min_array = np.min(array)
        min_for_dtype = np.iinfo(target_dtype).min
        assert min_array >= min_for_dtype, \
            'Broaden {} dtype as cast will underflow: {} < dtype limit {}'.\
                format(dtype_name, min_array, min_for_dtype)

        max_array = np.max(array)
        max_for_dtype = np.iinfo(target_dtype).max
        assert max_array <= max_for_dtype, \
            'Broaden {} dtype as cast will overflow {} > dtype limit {}'.\
                format(dtype_name, max_array, max_for_dtype)

    return np.array(array, dtype=target_dtype, copy=False)


Face = namedtuple('Face',
                  ['i1', 'i2', 'i3',
                   'n1', 'n2', 'n3',
                   'uv1', 'uv2', 'uv3', 'fid'
                   ])
                   # ], verbose=False)


class FaceGroup(object):

    def __init__(self):
        self.smooth = 0
        self.material_name = None
        self.group_name = None
        self.object_name = None


# .OBJ File Regular Expressions

# v float float float
vertex_pattern = re.compile(
    r'v( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)')

# vn float float float
normal_pattern = re.compile(
    r'vn( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)')

# vt float float
uv_pattern = re.compile(r'vt( +[\-\w|\d|\.|\+]+)( +[\-\w|\d|\.|\+]+)')

# f vertex vertex vertex ...
face_pattern1 = re.compile(r'f( +\d+)( +\d+)( +\d+)( +\d+)?')

# f vertex/uv vertex/uv vertex/uv ...
face_pattern2 = re.compile(
    r'f( +(\d+)\/(\d+))( +(\d+)\/(\d+))( +(\d+)\/(\d+))( +(\d+)\/(\d+))?')

# f vertex/uv/normal vertex/uv/normal vertex/uv/normal ...
face_pattern3 = re.compile(
    r'f( +(\d+)\/(\d+)\/(\d+))( +(\d+)\/(\d+)\/(\d+))( +(\d+)\/(\d+)\/(\d+))( +(\d+)\/(\d+)\/(\d+))?')

# f vertex//normal vertex//normal vertex//normal ...
face_pattern4 = re.compile(
    r'f( +(\d+)\/\/(\d+))( +(\d+)\/\/(\d+))( +(\d+)\/\/(\d+))( +(\d+)\/\/(\d+))?')


# .MTL File patterns
new_mtl_pattern = re.compile(r'newmtl (.+)')

# Kd 1.00 1.00 1.00
k_pattern = re.compile(
    r'(Kd|Ka|Ks)( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)( +[\d|\.|\+|\-|e]+)')

# Ns 1.00
single_pattern = re.compile(r'(Ns|Ni|d|illum)( +[\d|\.|\+|\-|e]+)')

# map_Kd blah.png
map_pattern = re.compile(
    r'(map_Ka|map_Kd|map_Ks|map_bump|bump|disp|map_Ns) (.+)')

# Can have optional arguments, but we're not parsing these ( e.g. map_Kd
# asdf.png -clamp on )
image_pattern = re.compile(r'(?:\w+) ([^-].+\.(tga|png|jpeg|jpg|bmp|tiff))')


class Texture(object):

    def __init__(self, image_name):
        self.image_name = image_name


class Material(object):

    def __init__(self, name=None):
        self.name = name


class Mesh(object):

    def __init__(self, material=None, deformation=None):
        self.material = material or Material()
        self.deformation = deformation

        # GL data structures
        # Should be np.array, len(verts) == len(faces)*3
        self.vertices = None
        self.indices = None

        self._parent_vertices = None

        self.faces = []         # [ [1,2,3], [4,5,6] ]
        self.face_normals = []  # [ [N1,N2,N3], [N1,N2,N3] ] or None
        self.face_uvs = []      # [ [UV1,UV2,UV3], [UV1,UV2,UV3] ] or None

    def compute_smooth_normals(self, max_smoothing_angle=25):
        """
        Computes normals for each vertex used by each face by averaging
        normals that are within 'max_smoothing_angle'.

        Many models have face indices defined for the front and back of a face,
        perhaps to ensure both are drawn. Blindly averaging all vertex normals
        together was resulting in near zero average vectors because of this.

        Warning: this is very slow.

        Parameters
        ----------
        max_smoothing_angle : float
            Cutoff for treating vectors as similar, in degrees.
        """
        if max_smoothing_angle < 0.0 or max_smoothing_angle > 180.0:
            raise ValueError("Expected max_smoothing_angle between 0 and 180")
        # We will use the cos of the angles to save time later on
        angle_thresh = max_smoothing_angle * np.pi / 180.0
        cos_thresh = math.cos(angle_thresh)

        vert_normal_groups = defaultdict(list)
        face_verts_to_groups = {}

        has_uvs = len(self.face_uvs) > 0

        if self.face_uvs:
            assert len(self.face_uvs) == len(self.faces), \
                "Expecting UVs for each face"

        # First filter out faces with invalid normals
        valid_faces = []
        valid_uvs = []
        face_normals = []
        face_normals_dict = {}

        for idx, face in enumerate(self.faces):
            i1, i2, i3, fid = face
            v1 = self._parent_vertices[i1]
            v2 = self._parent_vertices[i2]
            v3 = self._parent_vertices[i3]
            face_normal = get_face_normal(v3, v2, v1)
            if face_normal is not None:
                valid_faces.append((i1, i2, i3, fid))
                face_normals.append(np.array(face_normal))
                face_normals_dict[face] = np.array(face_normal)
                if has_uvs:
                    valid_uvs.append(self.face_uvs[idx])

        self.faces = valid_faces
        self.face_uvs = valid_uvs

        # -----------------------------
        # Create face mapping -- not optimized
        # -----------------------------
        # create a list of adjacent faces
        adjacent_faces = defaultdict(list)
        for face in self.faces:
            i1, i2, i3, fid = face
            adjacent_faces[i1].append(face)
            adjacent_faces[i2].append(face)
            adjacent_faces[i3].append(face)

        # map similar (in terms of normal) and nearby faces to the same fid
        new_fid_mapping = {}
        current_new_fid = 0

        def df_add_faces(new_fid, face, normal):
            i1, i2, i3, fid = face

            if fid in new_fid_mapping:
                return
            else:
                # check to see that their normals aren't too disparate

                # The face_normal vectors were constructed to have
                # norm 1, so u dot v is the cosine between u & v
                # Comparing the cosine is much faster than comparing
                # the actual angles.
                cos = face_normals_dict[face].dot(normal)
                if cos > cos_thresh:

                    new_fid_mapping[fid] = new_fid
                    for vert in [i1, i2, i3]:
                        for n in adjacent_faces[vert]:
                            df_add_faces(new_fid, n, face_normals_dict[face])

        for face in self.faces:
            i1, i2, i3, fid = face

            if fid in new_fid_mapping:
                continue
            else:
                current_new_fid += 1
                new_fid_mapping[fid] = current_new_fid

                if len(self.faces) > 1000:
                    continue

                for vert in [i1, i2, i3]:
                    for n in adjacent_faces[vert]:
                        df_add_faces(current_new_fid, n, face_normals_dict[face])
        self.num_faces = current_new_fid

        if self.deformation is not None:
            for i, v in enumerate(self._parent_vertices):
                self._parent_vertices[i] = self.deformation.dot(v)

            # Recompute normals after deformation
            face_normals = []
            face_normals_dict = {}

            for idx, face in enumerate(self.faces):
                i1, i2, i3, fid = face
                v1 = self._parent_vertices[i1]
                v2 = self._parent_vertices[i2]
                v3 = self._parent_vertices[i3]

                face_normal = get_face_normal(v3, v2, v1)
                if face_normal is not None:
                    face_normals.append(np.array(face_normal))
                    face_normals_dict[face] = np.array(face_normal)

        # Go through each vertex used by each face, if the face normal is
        # within 'angle_thresh' of a normal used by a different face, add
        # it to a group of nearby normals. Once all groups have been created,
        # compute the average of each group.
        # TODO: this can be rewritten to be significantly faster.
        # For example, you could store vectors normalized to be length 1,
        # keeping those vectors in a kdtree.  Then you can search in the
        # kdtree for the closest vector instead of iterating over all
        # vectors to find the closest.
        for idx, face in enumerate(self.faces):
            i1, i2, i3, fid = face
            v1 = self._parent_vertices[i1]
            v2 = self._parent_vertices[i2]
            v3 = self._parent_vertices[i3]

            face_normal = face_normals[idx]

            for vert_idx in [i1, i2, i3]:
                min_cos = -2.0 # basically -inf
                min_group = None

                for normal_group in vert_normal_groups[vert_idx]:
                    for normal in normal_group:
                        # The face_normal vectors were constructed to have
                        # norm 1, so u dot v is the cosine between u & v
                        # Comparing the cosine is much faster than comparing
                        # the actual angles.
                        cos = face_normal.dot(normal)
                        if cos > cos_thresh and cos > min_cos:
                            min_cos = cos
                            min_group = normal_group

                if min_group is not None:
                    min_group.append(face_normal)
                    face_verts_to_groups[(i1, i2, i3, vert_idx)] = min_group
                else:
                    new_group = [face_normal, ]
                    face_verts_to_groups[(i1, i2, i3, vert_idx)] = new_group
                    vert_normal_groups[vert_idx].append(new_group)  # new group

        # Compute the average vector for each group
        group_to_avg_normal = {}
        for vert, normal_groups in vert_normal_groups.items():
            for group in normal_groups:
                group_to_avg_normal[id(group)] = average_vectors(group)

        vertices = []
        normals = []
        indices = []
        uvs = []
        vertex_face_ids = []

        # Create the final list of vertices for each face, where each vertex
        # uses the average vector from the group it was assigned to.
        for idx, face in enumerate(self.faces):
            i1, i2, i3, fid = face
            fid = new_fid_mapping[fid] / 255.
            for vert_idx in [i1, i2, i3]:
                group = face_verts_to_groups[(i1, i2, i3, vert_idx)]
                avg_normal = group_to_avg_normal[id(group)]

                vertices.append(self._parent_vertices[vert_idx])
                normals.append(avg_normal)
                vertex_face_ids.append([fid, fid, fid])

            if has_uvs:
                uv1, uv2, uv3 = self.face_uvs[idx]
                uvs.append(uv1)
                uvs.append(uv2)
                uvs.append(uv3)

        indices = range(len(vertices))

        assert len(normals) == len(vertices)

        if len(vertices) == 0:
            self.vertices = None
            self.indices = None
            logger.warning("Empty mesh found")
            return

        vtype = [('a_position', np.float32, 3),
                 ('a_normal', np.float32, 3),
                 ('face_ids', np.float32, 3)]
        if has_uvs:
            vtype.append(('a_texcoord', np.float32, 2))

        gl_vertices = np.zeros(len(vertices), vtype)
        gl_vertices['a_position'] = vertices
        gl_vertices['a_normal'] = normals
        gl_vertices['face_ids'] = vertex_face_ids

        if has_uvs:
            gl_vertices['a_texcoord'] = uvs

        self.vertices = gl_vertices
        self.indices = np.array(indices)
        self.indices = checked_dtype_narrowing(self.indices, np.uint32)


def average_vectors(vectors):
    avg_vec = [0.0, 0.0, 0.0]
    for v1, v2, v3 in vectors:
        avg_vec[0] += v1
        avg_vec[1] += v2
        avg_vec[2] += v3

    avg_vec[0] /= float(len(vectors))
    avg_vec[1] /= float(len(vectors))
    avg_vec[2] /= float(len(vectors))
    return avg_vec


def get_face_normal(v1, v2, v3):
    # numpy is slow here
    a0 = v3[0] - v1[0]
    a1 = v3[1] - v1[1]
    a2 = v3[2] - v1[2]

    b0 = v2[0] - v1[0]
    b1 = v2[1] - v1[1]
    b2 = v2[2] - v1[2]

    n0 = a1 * b2 - a2 * b1
    n1 = a2 * b0 - a0 * b2
    n2 = a0 * b1 - a1 * b0

    mag = math.sqrt(n0 ** 2 + n1 ** 2 + n2 ** 2)

    if mag == 0:
        return None

    return [n0 / mag,
            n1 / mag,
            n2 / mag]


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    n1 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]
    n2 = v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]
    if n1 == 0.0 or n2 == 0.0:
        return np.pi / 2.0
    dot = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / math.sqrt(n1 * n2)
    if dot < -1.0:
        angle = np.pi
    elif dot > 1.0:
        angle = 0
    else:
        angle = np.arccos(dot)
    return angle


class MTLFile(object):

    def __init__(self, path):
        self.materials = []
        self.path = path
        self.load()

    def _match_line(self, line, pattern):
        """ Match 'pattern' to line, cache result and return match """
        self.match_result = pattern.match(line)
        return self.match_result

    def load(self):

        with open(self.path, 'r') as f:

            def float_result(idx):
                s = self.match_result.group(idx)
                return float(s) if s else None

            def int_result(idx):
                s = self.match_result.group(idx)
                return float(s) if s else None

            for line in f:
                line = line.strip()

                # Line comment
                if not line or line[0] == '#':
                    continue

                # newmtl material_name
                elif self._match_line(line, new_mtl_pattern):
                    res = self.match_result
                    mtl_name = res.group(1).strip()

                    mat = Material(name=mtl_name)
                    self.materials.append(mat)
                    self.current_material = mat

                # Kd, Ka, Ks - Diffuse/Ambient/Specular
                elif self._match_line(line, k_pattern):
                    k = self.match_result.group(1)
                    color = [float_result(2), float_result(3), float_result(4)]
                    key = {'Ka': 'ambient', 'Kd': 'diffuse', 'Ks': 'specular'}
                    setattr(self.current_material, key[k], color)

                elif self._match_line(line, single_pattern):
                    name = self.match_result.group(1)
                    mat = self.current_material

                    if name == 'Ns':
                        mat.specular_exp = float_result(2)
                    elif name == 'Ni':
                        # Don't know what this is
                        mat.Ni = float_result(2)
                    elif name == 'd' or name.lower() == 'tr':
                        mat.dissolve = float_result(2)
                    elif name == 'illum':
                        mat.illum = int_result(2)

                elif self._match_line(line, map_pattern):

                    mat = self.current_material
                    map_name = self.match_result.group(1)

                    # Only map_Kd supported currently
                    if map_name == 'map_Kd':

                        img_match = image_pattern.search(line)
                        if img_match:
                            image_name = img_match.group(1).strip()
                            mat.diffuse_texture = Texture(image_name)

                        # Optional clamp argument
                        clamp = re.search(r'-clamp (on|off)', line)
                        if clamp:
                            mat.diffuse_texture.clamp = clamp.group(1)

                    else:
                        logger.warning("Material discarding '{}'"
                                       .format(map_name))


class OBJFile(object):

    def __init__(self, path, deformation):
        self.path = path
        self.face_groups = defaultdict(FaceGroup)
        self.fid = 0
        self.deformation = deformation

        self.vertices = []
        self.normals = []
        self.uvs = []

        self.faces = []
        self.face_idx = 0

        self.materials = []
        self.materials_dict = {}
        self.current_material_name = None

        self.load()

    #######
    def _match_line(self, line, pattern):
        """ Match 'pattern' to line, cache result and return match """
        self.match_result = pattern.match(line)
        return self.match_result

    def _handle_vertex(self, result):
        vx = float(result.group(1))
        vy = float(result.group(2))
        vz = float(result.group(3))
        self.vertices.append([vx, vy, vz])

    def _handle_vertex_normal(self, result):
        vx = float(result.group(1))
        vy = float(result.group(2))
        vz = float(result.group(3))
        self.normals.append([vx, vy, vz])

    def _handle_uv(self, result):
        u = float(result.group(1))
        v = float(result.group(2))
        self.uvs.append([u, v])

    def _add_face(self, face, uv_idxs, normal_idxs):

        # Triangle face:
        if face[3] is None:
            i1 = face[0] - 1
            i2 = face[1] - 1
            i3 = face[2] - 1
            if normal_idxs:
                n1 = self.normals[normal_idxs[0] - 1]
                n2 = self.normals[normal_idxs[1] - 1]
                n3 = self.normals[normal_idxs[2] - 1]
            else:
                n1 = n2 = n3 = None

            if uv_idxs:
                uv1 = self.uvs[uv_idxs[0] - 1]
                uv2 = self.uvs[uv_idxs[1] - 1]
                uv3 = self.uvs[uv_idxs[2] - 1]
            else:
                uv1 = uv2 = uv3 = None

            self.faces.append(Face(i1, i2, i3, n1, n2, n3, uv1, uv2, uv3, self.fid))

        # Quad face:
        else:
            # Make two triangles
            for idx1, idx2, idx3 in ([0, 1, 3], [1, 2, 3]):

                i1 = face[idx1] - 1
                i2 = face[idx2] - 1
                i3 = face[idx3] - 1

                if normal_idxs:
                    n1 = self.normals[normal_idxs[idx1] - 1]
                    n2 = self.normals[normal_idxs[idx2] - 1]
                    n3 = self.normals[normal_idxs[idx3] - 1]
                else:
                    n1 = n2 = n3 = None

                if uv_idxs:
                    uv1 = self.uvs[uv_idxs[idx1] - 1]
                    uv2 = self.uvs[uv_idxs[idx2] - 1]
                    uv3 = self.uvs[uv_idxs[idx3] - 1]
                else:
                    uv1 = uv2 = uv3 = None

                self.faces.append(
                    Face(i1, i2, i3, n1, n2, n3, uv1, uv2, uv3, self.fid))

        self.fid += 1
        self.face_idx = len(self.faces)

    ## Material related ##
    def _handle_smooth(self, smooth):
        # TODO
        logger.debug("Discarding smoothing")

    def _handle_usemtl(self, mat_name):
        self.face_groups[self.face_idx].material_name = mat_name
        self.current_material_name = mat_name

    def _parse_mtl_file(self, mtl_file):
        logger.debug("Parsing MTL file {}".format(mtl_file))
        mtl_file = os.path.join(os.path.dirname(self.path), mtl_file)
        self.materials = MTLFile(mtl_file).materials
        self.materials_dict = {mat.name: mat for mat in self.materials}
        logger.debug(" parsed {} materials ".format(len(self.materials)))

    def _handle_new_group(self, group_name, file_obj):
        logger.debug(" Adding group: {}".format(group_name))
        self.face_groups[self.face_idx].group_name = group_name
        # New groups inherit current material unless overwritten later
        self.face_groups[self.face_idx].material_name = self.current_material_name

    def _start_new_object(self, obj_name=None):
        logger.debug(" Adding object: {}".format(obj_name))
        self.face_groups[self.face_idx].object_name = obj_name
        # New objects inherit current material unless overwritten later
        self.face_groups[self.face_idx].material_name = self.current_material_name

    #### Load ####
    def load(self):

        f = open(self.path, 'r')

        def result(idx):
            s = self.match_result.group(idx)
            return int(s) if s else None

        for line in f.readlines():
            line = line.strip()

            # Line comment
            if not line or line[0] == '#':
                continue

            # v float float float
            elif self._match_line(line, vertex_pattern):
                self._handle_vertex(self.match_result)

            # vn float float float
            elif self._match_line(line, normal_pattern):
                self._handle_vertex_normal(self.match_result)

            # vt float float
            elif self._match_line(line, uv_pattern):
                self._handle_uv(self.match_result)

            # f vertex vertex vertex ...
            elif self._match_line(line, face_pattern1):

                self._add_face(
                    [result(1), result(2), result(3), result(4)],  # faces
                    None,  # uv
                    None,  # normal
                )

            # f vertex/uv vertex/uv vertex/uv ...
            elif self._match_line(line, face_pattern2):

                self._add_face(
                    [result(2), result(5), result(8), result(11)],  # faces
                    [result(3), result(6), result(9), result(12)],  # uv
                    None,  # normal
                )

            # f vertex/uv/normal vertex/uv/normal vertex/uv/normal ...
            elif self._match_line(line, face_pattern3):

                self._add_face(
                    [result(2), result(6), result(10), result(14)],  # faces
                    [result(3), result(7), result(11), result(15)],  # uvs
                    [result(4), result(8), result(12), result(16)]  # normal
                )

            # f vertex//normal vertex//normal vertex//normal ...
            elif self._match_line(line, face_pattern4):

                self._add_face(
                    [result(2), result(5), result(8), result(11)],  # faces
                    None,  # uv
                    [result(3), result(6), result(9), result(12)]  # normal
                )

            # New Object
            elif re.match(r'^o ', line):
                obj_name = line[2:].strip()
                self._start_new_object(obj_name)

            # New Group
            elif re.match(r'^g', line):
                self._handle_new_group(line[2:].strip(), f)

            # use material
            elif re.match(r'^usemtl ', line):
                mat_name = line[7:].strip()
                self._handle_usemtl(mat_name)

            # Material definition
            elif re.match(r'^mtllib ', line):
                mtl_file = line[7:].strip()
                self._parse_mtl_file(mtl_file)

            # Smoothing
            elif re.match(r'^s ', line):
                smooth = line[2:].strip()
                self._handle_smooth(smooth)

        self._finalize_data()

    def _finalize_data(self):
        unique_face_groups = defaultdict(list)

        def face_group_hash(mod):
            return str(mod.material_name) + str(mod.has_uvs)

        face_idxs = sorted(self.face_groups.keys())

        num_faces = len(self.faces)
        for begin_idx, end_idx in zip(face_idxs, face_idxs[1:] + [num_faces, ]):
            if begin_idx >= num_faces:
                continue
            face_group = self.face_groups[begin_idx]
            face_group.face_begin_idx = begin_idx
            face_group.face_end_idx = end_idx
            face_group.has_uvs = (self.faces[begin_idx].uv1 is not None)
            unique_face_groups[face_group_hash(face_group)].append(face_group)

        meshes = []
        # Now create meshes for each renderable set of faces
        for group_hash, mergeable_groups in unique_face_groups.items():
            mergeable_groups.sort(key=lambda x: x.face_begin_idx)

            material_name = mergeable_groups[0].material_name
            material = self.materials_dict.get(material_name, None)
            mesh = Mesh(material=material, deformation=self.deformation)
            mesh._parent_vertices = self.vertices

            for face_group in mergeable_groups:
                for idx in range(face_group.face_begin_idx, face_group.face_end_idx):
                    face = self.faces[idx]
                    mesh.faces.append((face[0], face[1], face[2], face[-1]))
                    if face[3] is not None:
                        mesh.face_normals.append(face[3:6])
                    if face[6] is not None:
                        mesh.face_uvs.append(face[6:9])
            meshes.append(mesh)

        self.meshes = meshes

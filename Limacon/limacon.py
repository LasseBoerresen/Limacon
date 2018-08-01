from bokeh.plotting import figure, show
from bokeh.palettes import viridis
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

"""
Create a set of curves, limacon, in the xy plane, with regular z increments and gradual rotatation around z, and 
gradual scale in xy. Around this curve, starting at a variable point, create n points defining the centers of the 
shells. A shell is one infinity curve, extruded towards another infinity curve. 5 shell center should be default. 
between each center point, create shell tip points. Between center and tip points, create flange points. 

For every layer going up, rotate the point_type list by 1, so centers shift      

The infinity curve is calculated to have tangents equal to the line on which their tangent point lies. Thus, the 
infinity curve is a function of 1 center point and 4 tangent points and a tangent point curvature value and a center
point curvature value. 

How should the shells get their pointy look?. Maybe the inner tanget points should be sucked towards the center of the
lamp, and the out tangent points should be pushed out a bit, then the 3d infinity curve will 

To create each 



Mounting each infinity shell.
Each shell center line up on a continuos splin. This splin could be printed with deep enough x-cutouts to mount each
shell. The shell number should be etched into each individual shell and the position on the ribcage as well. If the 
ribcage is printed with appropriate tolerances, each shell should stay there snugly.   
"""


def limacon(theta, r1=0.25, r2=0.05, n=4, h=0, scale_shift=0, scale_value=0.1, scale_pure=1):
    """

    :param np.array theta: angle, radians
    :param float r1: base radius
    :param float r2: radius of dents
    :param int n: number of dents
    :param float h: height
    :param float scale_shift:
    :param float scale_value:
    :param float scale_pure:
    :return: matrix with rows of xyz coordinates.
    :rtype: np.ndarray
    """

    phi = phi_from_h(h)

    scale = np.e ** (-((h - 0.25 + scale_shift) ** 2) / scale_value)

    scale = scale * scale_pure

    r = r1 + r2 * np.cos(n * (theta - phi))

    x = r * np.cos(theta) * scale
    y = r * np.sin(theta) * scale
    z = np.ones(len(theta))*h

    return np.array([x, y, z]).transpose()


# rotation of the limacon curve reltive to height.
def phi_from_h(h, phi_min=0, phi_max=math.tau * 4 / 12, skew=False, tanh_skew=0.5):
    """


    :param h: height from 0 to 1
    :param phi_min: rotation start
    :param phi_max: rotation end
    :param bool skew: if true use tanh h skew of rotation, instead of linear
    :param float tanh_skew: skew range of tahn in radians
    :return: rotation as dependent on height.
    """

    if skew:
        phi_skew = (1 + np.tanh((h - 0.5) * math.tau * tanh_skew)) / 2
        phi = (phi_min + phi_skew * phi_max)
    else:
        phi = (phi_min + h * phi_max)

    return phi


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    step = math.tau / 1024
    theta = np.arange(0, math.tau + step, step)

    plot_triangles = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_levels = 12
    phi_min = 0
    phi_max = math.tau * 4 / 12
    n_shells = 5
    phi_shells = 0  # -math.tau/5
    step = math.tau / n_shells
    wall_thickness = 0.002  # everything is in meters.
    # moving control points towards center of half-shell, makes flat shells closer together, therefore extra scale.
    wall_thickness_extra_scale = 2.0
    n_triangles = 256  # 8192
    # n_triangles = 8192
    scale_from_mm_to_m = 1000
    pointiness = -0.04  # in meters

    sides = []
    sides_wireframe = []
    for s, (side_scale, h_min, h_max, pure_scale_dir) in enumerate([(1.0, 0.0, 0.5, 1), (0.5, -0.025, 0.4, -1)]):
        centers = np.arange(phi_shells, math.tau + phi_shells, step)
        inner_left = centers - (step / 8) * 1
        inner_right = centers + (step / 8) * 1
        centers_left = centers - (step / 8) * 2
        centers_right = centers + (step / 8) * 2
        outer_left = centers - (step / 8) * 3
        outer_right = centers + (step / 8) * 3
        tips_left = centers - (step / 8) * 4
        tips_right = centers + (step / 8) * 4

        shells_list = []
        shells_wireframe_list = []

        h_step = (h_max - h_min) / n_levels
        for i, h in enumerate(np.arange(h_min, h_max, h_step)):

            layers_xyz = limacon(theta, h=h, scale_pure=side_scale)
            ax.plot(xs=layers_xyz[:, 0], ys=layers_xyz[:, 1], zs=layers_xyz[:, 2], color='#' + hex(int(0xaaaaaa * side_scale))[2:])
            # layers_xyz = limacon(theta, h=h + h_step / 2, scale_pure=side_scale)
            # ax.plot(xs=layers_xyz[:, 0], ys=layers_xyz[:, 1], zs=layers_xyz[:, 2], color='#' + hex(int(0xaaaaaa * side_scale))[2:])
            # layers_xyz = limacon(theta, h=h - h_step / 2, scale_pure=side_scale)
            # ax.plot(xs=layers_xyz[:, 0], ys=layers_xyz[:, 1], zs=layers_xyz[:, 2], color='#' + hex(int(0xaaaaaa * side_scale))[2:])


            # x_tips, y_tips = limacon(just_the_tips, scale=scale)
            # ax.scatter(xs=x_tips, ys=y_tips, zs=h, c='#aa0000', marker='o')

            center_pure_scale = side_scale * 1.0  # * 1.0 -0.2 * pure_scale_dir
            inner_pure_scale = side_scale * 1.0  # * 1.0 - 0.15 * pure_scale_dir
            elbow_pure_scale = side_scale * 1.0  # * 1.0 - 0.1 * pure_scale_dir
            outer_pure_scale = side_scale * 1.0  # * 1.0 - 0.05 * pure_scale_dir
            tips_pure_scale = side_scale * 1.0  # * 1.0 - 0.0 * pure_scale_dir

            centers_xyz = limacon(centers, h=h,scale_pure=center_pure_scale)
            inner_left_upper_xyz = limacon(inner_left, h=h + h_step / 2, scale_pure=inner_pure_scale)
            elbow_left_upper_xyz = limacon(centers_left, h=h + h_step, scale_pure=elbow_pure_scale)
            outer_left_upper_xyz = limacon(outer_left, h=h + h_step / 2, scale_pure=outer_pure_scale)
            tips_left_xyz = limacon(tips_left, h=h, scale_pure=tips_pure_scale)
            outer_left_lower_xyz = limacon(outer_left, h=h - h_step / 2, scale_pure=outer_pure_scale)
            elbow_left_lower_xyz = limacon(centers_left, h=h - h_step, scale_pure=elbow_pure_scale)
            inner_left_lower_xyz = limacon(inner_left, h=h - h_step / 2, scale_pure=inner_pure_scale)
            centers_xyz = limacon(centers, h=h, scale_pure=center_pure_scale)
            inner_right_upper_xyz = limacon(inner_right, h=h + h_step / 2, scale_pure=inner_pure_scale)
            elbow_right_upper_xyz = limacon(centers_right, h=h + h_step, scale_pure=elbow_pure_scale)
            outer_right_upper_xyz = limacon(outer_right, h=h + h_step / 2, scale_pure=outer_pure_scale)
            tips_right_xyz = limacon(tips_right, h=h, scale_pure=tips_pure_scale)
            outer_right_lower_xyz = limacon(outer_right, h=h - h_step / 2, scale_pure=outer_pure_scale)
            elbow_right_lower_xyz = limacon(centers_right, h=h - h_step, scale_pure=elbow_pure_scale)
            inner_right_lower_xyz = limacon(inner_right, h=h - h_step / 2, scale_pure=inner_pure_scale)

            # ax.scatter(xs=centers_xyz[:, 0], ys=centers_xyz[:, 1], zs=centers_xyz[:, 2], c='#ff0000', marker='o')

            # [point-type along line, shell, corrdinates_xyz]
            shells = np.array([
                centers_xyz,
                inner_left_upper_xyz,
                elbow_left_upper_xyz,
                outer_left_upper_xyz,
                tips_left_xyz,
                outer_left_lower_xyz,
                elbow_left_lower_xyz,
                inner_left_lower_xyz,
                centers_xyz,
                inner_right_upper_xyz,
                elbow_right_upper_xyz,
                outer_right_upper_xyz,
                tips_right_xyz,
                outer_right_lower_xyz,
                elbow_right_lower_xyz,
                inner_right_lower_xyz,
            ])

            # move axes so first dim is shell number, next is point type and last is xyz coordinates.
            shells = shells.transpose((1, 0, 2))

            # move all points except centers towards mean of half-shells

            for shell_idx, shell in enumerate(shells):
                half_shell_len = int(len(shells[shell_idx]) / 2)

                # Do first half of shell
                half_shell = shells[shell_idx][1:half_shell_len]
                mean_tmp = np.mean(half_shell, axis=0)  # 1:... to exclude center
                diff_vectors = mean_tmp - half_shell

                # find a scale to each diff vectors length to match that for wall thickness/2
                # .../2 to scale only half wall thickness
                diff_vectors_scale = (wall_thickness * wall_thickness_extra_scale/np.linalg.norm(diff_vectors, axis=1)) / 2
                for v, vec in enumerate(diff_vectors):
                    diff_vectors[v] *= diff_vectors_scale[v]
                shells[shell_idx][1:half_shell_len] += diff_vectors

                # Do the other half of the shell. Could be done as a loop to not repeat code
                half_shell = shells[shell_idx][half_shell_len + 1:]
                mean_tmp = np.mean(half_shell, axis=0)  # 1:... to exclude center
                diff_vectors = mean_tmp - half_shell
                # find a scale to each diff vectors length to match that for wall thickness/2
                # .../2 to scale only half wall thickness
                diff_vectors_scale = (wall_thickness * wall_thickness_extra_scale / np.linalg.norm(diff_vectors, axis=1)) / 2
                for v, vec in enumerate(diff_vectors):
                    diff_vectors[v] *= diff_vectors_scale[v]
                shells[shell_idx][half_shell_len + 1:] += diff_vectors



            # add points to fit tangent.
            tan_delta = 0.15
            shells_tan = []
            shells_wireframe = []
            for i, shell in enumerate(shells):
                shell_tan = []
                shell_wireframe = []
                for j, point in enumerate(shell):
                    # Only add first, halfway, which is the center of the infinity curve
                    shell_wireframe.append(point)
                    if j == 0 or j == len(shell) / 2:
                        shell_tan.append(point)
                    elif not j % 2 == 0:
                        point_diff_forwards = shell[(j + 1) % len(shell)] - shell[j-1]
                        point_diff_backwards = shell[j - 1] - shell[(j + 1) % len(shell)]
                        shell_tan.append(point + point_diff_backwards * tan_delta)
                        shell_tan.append(point)
                        shell_tan.append(point + point_diff_forwards * tan_delta)
                shells_tan.append(np.array(shell_tan))
                shells_wireframe.append(np.array(shell_wireframe))
            shells = np.array(shells_tan)
            shells_wireframe = np.array(shells_wireframe)

            shells_list.append(shells_tan)
            shells_wireframe_list.append(shells_wireframe)

            centers += step / 4
            inner_left += step / 4
            inner_right += step / 4
            centers_left += step / 4
            centers_right += step / 4
            outer_left += step / 4
            outer_right += step / 4
            tips_left += step / 4
            tips_right += step / 4

        sides.append(shells_list)
        sides_wireframe.append(shells_wireframe_list)


        # for i, shells_layer in enumerate(shells_wireframe_list):
        #     for j, shell in enumerate(shells_layer):
        #         ax.plot(xs=shell[:, 0], ys=shell[:, 1], zs=shell[:, 2], color='green')
        #         break
        # #
        # for i, shells_layer in enumerate(shells_list):
        #     for j, shell in enumerate(shells_layer):
        #         ax.plot(xs=shell[:, 0], ys=shell[:, 1], zs=shell[:, 2], color='blue')
        #         # ax.scatter(xs=shell[:, 0], ys=shell[:, 1], zs=shell[:, 2], color='blue')
        #         break
        #
        # for i, shells_layer in enumerate(shells_list):
        #     for j, shell in enumerate(shells_layer):
        #         shell_T = shell.transpose()
        #         tck, u = splprep([shell_T[0], shell_T[1], shell_T[2]], per=True, s=0.0, k=3)
        #         unew = np.arange(0, 1.01, 0.01)
        #         out = splev(unew, tck)
        #         ax.plot(xs=out[0], ys=out[1], zs=out[2], color='#' + hex(int(0xff * side_scale))[2:]+'0000')
        #         if j == 1:
        #             break
        #             pass

    # Generate Triangles
    for i, shells_layer in enumerate(shells_list):
        for j, shell in enumerate(shells_layer):
            tck_0, u_0 = splprep(
                [sides[0][i][j].transpose()[0], sides[0][i][j].transpose()[1], sides[0][i][j].transpose()[2]], per=True,
                s=0.0, k=3)
            tck_1, u_1 = splprep(
                [sides[1][i][j].transpose()[0], sides[1][i][j].transpose()[1], sides[1][i][j].transpose()[2]], per=True,
                s=0.0, k=3)
            # Generate evenly spaced values of the parametric variable t, from start to end of each half-shell
            u_0_new = np.linspace(0, 1, int(n_triangles/8), endpoint=False)
            u_1_new = np.linspace(0, 1, int(n_triangles/8), endpoint=False)

            # Evalueate the shell functions for each half-shell
            out_0 = np.array(splev(u_0_new, tck_0))
            out_1 = np.array(splev(u_1_new, tck_1))

            out = []

            # interlace points of the two curves
            triangles = []
            n_points = len(out_0[0])

            # make tips more pointy

            for k in range(n_points):
                inside_to_out_diff_vectors = out_0[:, k] - out_1[:, k]
                inside_to_out_diff_vectors_unit = inside_to_out_diff_vectors / np.linalg.norm(
                    inside_to_out_diff_vectors)
                out_0[:, k] = out_0[:, k] + inside_to_out_diff_vectors_unit * np.abs(np.cos(k / n_points * math.tau)) * pointiness
                # out_1[:, k] = out_1[:, k] + inside_to_out_diff_vectors_unit * np.abs(np.cos(k / n_points * math.tau)) * pointiness

            for k in range(n_points):
                # Triangles defined clockwise, i.e. top surface outwards.
                k_next = (k + 1) % n_points

                triangles.append(np.array([out_0[:, k], out_0[:, k_next], out_1[:, k_next]]))

                assert not np.array_equal(triangles[-1][0], triangles[-1][1])
                assert not np.array_equal(triangles[-1][0], triangles[-1][2])
                assert not np.array_equal(triangles[-1][1], triangles[-1][2])

                triangles.append(np.array([out_1[:, k], out_0[:, k], out_1[:, k_next]]))

                assert not np.array_equal(triangles[-1][0], triangles[-1][1])
                assert not np.array_equal(triangles[-1][0], triangles[-1][2])
                assert not np.array_equal(triangles[-1][1], triangles[-1][2])

            triangles = np.array(triangles)

            def rotate_around_axis(vec: np.ndarray, u: np.ndarray, a: float):
                """
                asdf

                :param vec:
                :param u:
                :param a:
                :return:
                """

                ux = u[0]
                uy = u[1]
                uz = u[2]
                ncos = 1 - np.cos(a)
                cosa = np.cos(a)
                sina = np.sin(a)

                rotation_mat = np.array([
                    [cosa+ux**2*ncos, ux*uy*ncos-uz*sina, ux*uz*ncos+uy*sina],
                    [uy*ux*ncos+uz*sina, cosa+uy**2*ncos, uy*uz*ncos-ux*sina],
                    [uz*ux*ncos-uy*sina, uz*uy*ncos+ux*sina, cosa+uz**2*ncos]
                ])

                return np.matmul(vec, rotation_mat)

            def unit_vector(vector):
                """ Returns the unit vector of the vector.  """
                return vector / np.linalg.norm(vector)

            def angle_between(v1, v2):
                """ Returns the angle in radians between vectors 'v1' and 'v2'::

                        >>> angle_between((1, 0, 0), (0, 1, 0))
                        1.5707963267948966
                        >>> angle_between((1, 0, 0), (1, 0, 0))
                        0.0
                        >>> angle_between((1, 0, 0), (-1, 0, 0))
                        3.141592653589793
                """
                v1_u = unit_vector(v1)
                v2_u = unit_vector(v2)
                return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

            # Calculate the rotation matrix that moves shell to be upright and aligned with x-axis.
            # find normal vector to z-vector and center-vector. Rotate around this normal vector the angle between
            # the two vectors. Boom. Then rotate around z angle between tip vector and x.

            # Z_vector of a shell is the vector from the last to the first point of the first triangle
            center_vec = triangles[0][0] - triangles[0][2]
            center_vec_unit = unit_vector(center_vec)
            z_vector = np.array([0, 0, 1])
            rotation_axis_vector = np.cross(z_vector, center_vec_unit)
            angle_diff = angle_between(z_vector, center_vec_unit)

            center_vec_rotated = rotate_around_axis(center_vec, rotation_axis_vector, angle_diff)

            # Create inner and outer shells perpendicular to triangles
            # Each orig triangle only defines 1 new point on the outer and inner shells. The new outer and inner
            # triangles will have to be found from the new single points, to make them match up.
            # s stands for shell, c stands for curve
            points_s_out_c_out = []
            points_s_inn_c_out = []
            points_s_out_c_inn = []
            points_s_inn_c_inn = []
            for tri_idx in range(int(len(triangles) / 2)):
                # Triangle indexes of outer and inner infinity loops
                tri_idx_out = tri_idx * 2
                tri_idx_inn = tri_idx * 2 + 1
                # norm_vec points up on triangles defined clockwise
                norm_vec_out = np.cross(
                    triangles[tri_idx_out][2] - triangles[tri_idx_out][0],
                    triangles[tri_idx_out][1] - triangles[tri_idx_out][0])
                norm_vec_out_unit = norm_vec_out / np.linalg.norm(norm_vec_out, 2)
                points_s_out_c_out.append(triangles[tri_idx_out][0] + norm_vec_out_unit * wall_thickness / 2)
                points_s_inn_c_out.append(triangles[tri_idx_out][0] - norm_vec_out_unit * wall_thickness / 2)

                norm_vec_inn = np.cross(
                    triangles[tri_idx_inn][2] - triangles[tri_idx_inn][0],
                    triangles[tri_idx_inn][1] - triangles[tri_idx_inn][0])
                norm_vec_inn_unit = norm_vec_inn / np.linalg.norm(norm_vec_inn, 2)
                points_s_out_c_inn.append(triangles[tri_idx_inn][0] + norm_vec_out_unit * wall_thickness / 2)
                points_s_inn_c_inn.append(triangles[tri_idx_inn][0] - norm_vec_out_unit * wall_thickness / 2)

            points_s_out_c_out = np.array(points_s_out_c_out)
            points_s_inn_c_out = np.array(points_s_inn_c_out)
            points_s_out_c_inn = np.array(points_s_out_c_inn)
            points_s_inn_c_inn = np.array(points_s_inn_c_inn)

            # create triangles of outer and inner shells.
            triangles_s_out = []
            triangles_s_inn = []
            triangles_c_out = []
            triangles_c_inn = []

            n_points = len(points_s_out_c_inn)
            for k in range(n_points):
                # Triangles defined clockwise, i.e. top surface outwards.
                k_next = (k + 1) % n_points
                triangles_s_out.append(np.array(
                    [points_s_out_c_out[k], points_s_out_c_out[k_next], points_s_out_c_inn[k_next]]))
                triangles_s_out.append(np.array([
                    points_s_out_c_inn[k], points_s_out_c_out[k], points_s_out_c_inn[k_next]]))

                triangles_s_inn.append(np.array(
                    [points_s_inn_c_out[k], points_s_inn_c_out[k_next], points_s_inn_c_inn[k_next]]))
                triangles_s_inn.append(np.array([
                    points_s_inn_c_inn[k], points_s_inn_c_out[k], points_s_inn_c_inn[k_next]]))

                # Edge/curve triangles
                # Outer edge/curve
                triangles_c_out.append(np.array(
                    [points_s_out_c_out[k], points_s_inn_c_out[k], points_s_out_c_out[k_next]]))
                triangles_c_out.append(np.array([
                    points_s_inn_c_out[k], points_s_out_c_out[k_next], points_s_inn_c_out[k_next]]))

                # Inner edge/curve
                triangles_c_inn.append(np.array(
                    [points_s_out_c_inn[k], points_s_inn_c_inn[k], points_s_out_c_inn[k_next]]))
                triangles_c_inn.append(np.array([
                    points_s_inn_c_inn[k], points_s_inn_c_inn[k_next], points_s_out_c_inn[k_next]]))

            triangles_s_out = np.array(triangles_s_out)
            triangles_s_inn = np.array(triangles_s_inn)
            triangles_c_out = np.array(triangles_c_out)
            triangles_c_inn = np.array(triangles_c_inn)

            # To rotate all triangle, each point in each triangle must be multiplied with the rotation matrix
            # The rotation matrix should rotate so vector from center_inn to center_out is z-axis and right to tip left
            # is x axis.
            # TODO LB 20180731: output .obj file with rotated triangles.

            # Rotate all points in triangles to have center vector up.
            for tris, triangles in enumerate((triangles_s_out, triangles_s_inn, triangles_c_out, triangles_c_inn)):
                for tri, triangle in enumerate(triangles):
                    for v, vertex in enumerate(triangle):
                        triangles[tri][v] = rotate_around_axis(triangles[tri][v], rotation_axis_vector, angle_diff)

            # add triangles to .obj file
            obj_file = ''

            for tri, triangles in enumerate((triangles_s_out, triangles_s_inn, triangles_c_out, triangles_c_inn)):
                for triangle in triangles:
                    for vertex in triangle:
                        obj_file += 'v'
                        for coord in vertex:
                            obj_file += ' ' + str(coord*scale_from_mm_to_m)
                        obj_file += '\n'

            for tri, triangles in enumerate((triangles_s_out, triangles_s_inn, triangles_c_out, triangles_c_inn)):
                obj_file += 'g ' + str(tri) + '\n'
                for t, triangle in enumerate(triangles):
                    obj_file += 'f'
                    for v, vertex in enumerate(triangle):
                        obj_file += ' ' + str(tri * len(triangles)*3 + t*3 + v + 1)
                    obj_file += '\n'

            with open('shell_layer_{i}_shell_{j}.obj'.format(i=i, j=j), 'w') as f:
                f.write(obj_file)

            for tri, triangles in enumerate((triangles_s_out, triangles_s_inn, triangles_c_out, triangles_c_inn)):

                # Plot triangles using matplotlib
                if plot_triangles:
                    triangle_patches = Poly3DCollection(triangles)

                    layer_color = i/len(shells_list)
                    layer_color_next = (i+1)/len(shells_list)
                    triangle_colors = np.linspace(layer_color, layer_color_next, len(triangles))

                    triangle_colors_list = np.array(
                        [triangle_colors, triangle_colors, triangle_colors, np.ones(triangle_colors.shape) * 1]).transpose()
                    triangle_patches.set_edgecolor(triangle_colors_list*0.95)
                    triangle_patches.set_facecolor(triangle_colors_list)
                    ax.add_collection3d(triangle_patches)



            if j == 1:
                # break
                pass
            # plt.show()
        if i == 2:
            # break
            pass




    # TODO LB 20180729: Redefine wireframe points to find infinity curves with a cirtain spacing to allow for shells to have a thickness.
    # TODO LB 20180729: Rotate each shell so the two curve center points lie on the x axis, for printing.

    plt.show()


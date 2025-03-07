"""

Author:
    Andrés Beltrán-Pulido

Revision (MM/DD/YYYY):
    03/05/2025
"""


import jax
jax.config.update("jax_enable_x64", True)

import os
import h5py

import numpy as np
import equinox as eqx
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt

from functools import partial
from matplotlib.tri import Triangulation

from jaxtyping import Array
from typing import NamedTuple
from jax import jit, vmap


class Mesh(NamedTuple):
    vertices : Array
    vertex_markers : Array
    vertices_mask : Array
    triangles : Array
    triangles_mask : Array
    segments : Array
    segment_markers : Array
    segments_mask : Array


def _rotate_mesh(mesh: Mesh, angle: float) -> Mesh:

    # unpack info from the mesh
    vertices = mesh.vertices
    vertex_markers = mesh.vertex_markers
    vertices_mask = mesh.vertices_mask
    triangles = mesh.triangles
    triangles_mask = mesh.triangles_mask
    segments = mesh.segments
    segment_markers = mesh.segment_markers
    segments_mask = mesh.segments_mask

    # rotation matrix
    R = jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                   [jnp.sin(angle),  jnp.cos(angle)]], dtype=jnp.float64)

    # new vertives
    vertices_new = jnp.dot(vertices, R.T)

    # mesh
    new_mesh = Mesh(vertices = jnp.array(vertices_new, dtype=jnp.float64),
                    vertex_markers = jnp.array(vertex_markers, dtype=jnp.int32),
                    vertices_mask = jnp.array(vertices_mask, dtype=jnp.bool_),
                    triangles = jnp.array(triangles, dtype=jnp.int32),
                    triangles_mask = jnp.array(triangles_mask, dtype=jnp.bool_),
                    segments = jnp.array(segments, dtype=jnp.int32),
                    segment_markers = jnp.array(segment_markers, dtype=jnp.int32),
                    segments_mask = jnp.array(segments_mask, dtype=jnp.bool_))
    
    return new_mesh


plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'Times New Roman'})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def plot_B(B, stator_mesh, rotor_mesh, air_gap_mesh):

    m2cm = 1.0e2

    # merge the meshes
    vertices = jnp.vstack([stator_mesh.vertices, rotor_mesh.vertices, air_gap_mesh.vertices])
    triangles = jnp.vstack([stator_mesh.triangles,
                            rotor_mesh.triangles + stator_mesh.vertices.shape[0],
                            air_gap_mesh.triangles + stator_mesh.vertices.shape[0] + rotor_mesh.vertices.shape[0]])

    # Create triangulation for ploting
    x = m2cm*vertices[:, 0]
    y = m2cm*vertices[:, 1]
    trian = triangles
    triangulation = Triangulation(x, y, trian)

    cax = plt.tripcolor(triangulation, B, edgecolors='None',
                        cmap=plt.cm.gray, vmin=0, vmax=2.2, linewidth=0.1)

    return cax


def zero_crossings(signal):
    # Ensure signal is a numpy array
    signal = np.array(signal)
    upper_threshold = np.max(np.abs(signal)) * 0.05 

    outputs = [1.0]
    output = -1.0

    # Process each signal in the list
    for input_signal in signal[1:]:
        if input_signal > upper_threshold:
            output = 1.0

        elif output == 1.0 and input_signal < 0.0:
            output = -1.0
      
        outputs.append(output)

    # Detecting crossings
    crossings = np.where(np.diff(np.sign(np.array(outputs))))[0]

    return crossings

def compute_bmax_bmin_stator(B, dBdt):

    idx_bmax = np.argmax(B)
    # roll signal to start at maximum
    B = np.roll(B, -idx_bmax)
    dBdt = np.roll(dBdt, -idx_bmax)

    # zero crossings
    crossings_first_half = zero_crossings(dBdt[:len(dBdt)//2])
    if (len(crossings_first_half) - 1) % 2 != 0:
        crossings_first_half = crossings_first_half[:-1]
    
    crossings_second_half = crossings_first_half + len(dBdt)//2
    crossings = np.concatenate([crossings_first_half, crossings_second_half])
    B_crossings = B[crossings]
    max_or_min = np.ones_like(B_crossings)
    for k in range(1, len(B_crossings)):
        max_or_min[k] = -max_or_min[k-1]
    
    bmaxl = []
    bminl = []
    crossmaxl = []
    crossminl = []

    while len(B_crossings) > 2:
        if max_or_min[0] == 1:
            
            if B_crossings[0] < B_crossings[2]:

                bmaxl.append(B_crossings[0])
                bminl.append(B_crossings[1])

                crossmaxl.append(crossings[0])
                crossminl.append(crossings[1])

                mask = np.ones_like(B_crossings, dtype=bool)
                mask[[0, 1]] = False
                B_crossings = B_crossings[mask]
                crossings = crossings[mask]
                max_or_min = max_or_min[mask]

            else:
                B_crossings = np.roll(B_crossings, -1)
                crossings = np.roll(crossings, -1)
                max_or_min = np.roll(max_or_min, -1)

        elif max_or_min[0] == -1:

            if B_crossings[0] > B_crossings[2]:
                bmaxl.append(B_crossings[1])
                bminl.append(B_crossings[0])

                crossmaxl.append(crossings[1])
                crossminl.append(crossings[0])

                mask = np.ones_like(B_crossings, dtype=bool)
                mask[[0, 1]] = False
                B_crossings = B_crossings[mask]
                crossings = crossings[mask]
                max_or_min = max_or_min[mask]

            else:
                B_crossings = np.roll(B_crossings, -1)
                crossings = np.roll(crossings, -1)
                max_or_min = np.roll(max_or_min, -1)
    
    if B_crossings[0] < B_crossings[1]:
        bmaxl.append(B_crossings[1])
        bminl.append(B_crossings[0])

        crossmaxl.append(crossings[1])
        crossminl.append(crossings[0])
        
    else:
        bmaxl.append(B_crossings[0])
        bminl.append(B_crossings[1])

        crossmaxl.append(crossings[0])
        crossminl.append(crossings[1])

    mask = np.ones_like(B_crossings, dtype=bool)
    mask[[0, 1]] = False
    B_crossings = B_crossings[mask]
    max_or_min = max_or_min[mask]

    return np.array(bmaxl), np.array(bminl), np.array(crossmaxl), np.array(crossminl)


def compute_stator_hysteresis_loss_component(B, dBdtheta_e, k_h, fe, num_teeth, axial_length, delta):

    ph_bar = np.zeros(B.shape[0])

    for ele in range(B.shape[0]):
    
        bmaxl_out, bminl_out, _, _ = compute_bmax_bmin_stator(B[ele], dBdtheta_e[ele])

        ellipse_bm = (bmaxl_out - bminl_out) / 2.0
        ellipse_hm = (1/np.pi)*k_h*ellipse_bm
        ellipse_areas = np.pi * ellipse_hm * ellipse_bm

        # hysteresis loss
        ph_bar[ele] = fe * np.sum(ellipse_areas) 

    ph = axial_length*(num_teeth/2)*np.sum(ph_bar * delta)

    return ph

def compute_stator_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_teeth, axial_length, delta):

    pc_x_bar = jnp.sum(ks**2 * Bx_max**2, axis=1)
    pc_y_bar = jnp.sum(ks**2 * By_max**2, axis=1)
    pc_bar = pc_x_bar + pc_y_bar

    pc = axial_length*(num_teeth/2)*k_c*(fe**2)*np.sum(pc_bar * delta)

    return pc

def compute_stator_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_teeth, axial_length, delta):
    
    Ce = 8.763363 
    ke_prime = k_e / Ce
    we = 2 * np.pi * fe

    pe_bar = jnp.mean((dBxdtheta_e**2 + dBydtheta_e**2)**0.75, axis=1)
    pe = axial_length*(num_teeth/2)*ke_prime*(we**1.5)*np.sum(pe_bar * delta)

    return pe


def q_of_element(element_vertices: Array) -> Array:
   
    y = element_vertices[:, 1]

    q1 = y[1] - y[2]
    q2 = y[2] - y[0]
    q3 = -(q1 + q2)  # 0 = q1 + q2 + q3

    return jnp.array([q1, q2, q3])

def r_of_element(element_vertices: Array) -> Array:

    x = element_vertices[:, 0]
    
    r1 = x[2] - x[1]
    r2 = x[0] - x[2]
    r3 = -(r1 + r2)  # 0 = r1 + r2 + r3

    return jnp.array([r1, r2, r3])

def area_of_element(element_vertices: Array) -> Array:

    r = r_of_element(element_vertices)
    q = q_of_element(element_vertices)

    dlt = 0.5 * (r[2] * q[1] - q[2] * r[1])

    return dlt

vec_dlt_func = jit(vmap(area_of_element, in_axes=(0,)))


def tooth_pair_centroids_and_deltas(mesh):

    n_tri_tp = mesh.triangles.shape[0] // 3

    # unpack from mesh
    vertices = mesh.vertices
    triangles = mesh.triangles[n_tri_tp:2*n_tri_tp]

    # the vertices of each element    
    elements_vertices = vertices[triangles]

    # centroids
    centroids = vec_xc_func(elements_vertices)
    delta = vec_dlt_func(elements_vertices)

    return centroids, delta

def compute_radial_and_tangetial_Bfield(B, mesh):
    
    # unpack from mesh
    vertices = mesh.vertices
    triangles = mesh.triangles 
    
    # the vertices of each element    
    elements_vertices = vertices[triangles]

    # centroids
    centroids = vec_xc_func(elements_vertices)

    def element_Bt_Br(Bx, By, centroid):

        # center of triangle
        Cx = centroid[0]
        Cy = centroid[1]

        # vector magnitude center of triangle
        Cmag = jnp.sqrt(Cx**2 + Cy**2)

        # unit radial and tangential (ccw) vectors
        arx = Cx/Cmag
        ary = Cy/Cmag
        atx = -ary
        aty = arx
        # radial component of B-field
        Br = Bx*arx + By*ary

        # tangential component of B-field
        Bt = Bx*atx + By*aty
      
        return Br, Bt

    vBt, vBr = jax.vmap(element_Bt_Br)(B[:, 0],
                                       B[:, 1],
                                       centroids)

    return vBt, vBr


def build_tooth_pair_Bfields(Bs, mesh, xi_geometry, xi_discrete):

    num_rotor_positions = Bs.shape[0]
    
    # allocate memory for radial and tangential B-fields
    Bt_s = np.zeros((num_rotor_positions,
                     mesh.triangles.shape[0]))
    
    Br_s = np.zeros((num_rotor_positions,
                     mesh.triangles.shape[0]))
   
    for i in range(num_rotor_positions):       
        # B-field of stator
        Bt_s[i, :], Br_s[i, :] = compute_radial_and_tangetial_Bfield(Bs[i], mesh)

    # number of triangles in tooth pair
    n_tri_tp = mesh.triangles.shape[0] // 3

    # reconstruct B-fields for each elements in a tooth pair
    Bt_tp = jnp.vstack(( Bt_s[:, 2*n_tri_tp:], 
                         Bt_s[:, n_tri_tp:2*n_tri_tp], 
                         Bt_s[:, :n_tri_tp],
                        -Bt_s[:, 2*n_tri_tp:],
                        -Bt_s[:, n_tri_tp:2*n_tri_tp],
                        -Bt_s[:, :n_tri_tp]))
    
    Br_tp = jnp.vstack(( Br_s[:, 2*n_tri_tp:],
                         Br_s[:, n_tri_tp:2*n_tri_tp],
                         Br_s[:, :n_tri_tp],
                        -Br_s[:, 2*n_tri_tp:],
                        -Br_s[:, n_tri_tp:2*n_tri_tp],
                        -Br_s[:, :n_tri_tp]))

    base_centroids, base_delta = tooth_pair_centroids_and_deltas(mesh)

    # map get Bx and By for each element in tooth pair for each rotor position in [0, 2*pi] electric radians
    Cx = base_centroids[:, 0]
    Cy = base_centroids[:, 1]
    Cmag = jnp.sqrt(Cx**2 + Cy**2)
    
    arx = Cx/Cmag
    ary = Cy/Cmag
    atx = -ary
    aty = arx

    _Bx = Br_tp*arx + Bt_tp*atx
    _By = Br_tp*ary + Bt_tp*aty

    # let us filter out the elements that are not in the core
    chi = vmap(indicator_core, in_axes=(0, None, None))(base_centroids, xi_geometry, xi_discrete)
    cond = chi > 0.5
    delta = base_delta[cond]
    Bx = _Bx[:, cond]
    By = _By[:, cond]

    return Bx.T, By.T, delta


def compute_stator_core_loss(Bs, mesh, xi_geometry, xi_material, xi_discrete):

    # unpack 
    axial_length = float(xi_geometry.axial_length)
    mechanical_speed = float(xi_geometry.mechanical_rotor_speed)
    num_poles = float(xi_discrete.number_of_poles)
    num_teeth = float(xi_discrete.number_of_slots)
    k_h = float(xi_material.hysteresis_loss_coefficient)
    k_c = float(xi_material.eddy_current_loss_coefficient)
    k_e = float(xi_material.excess_loss_coefficient)

    # compute electrical frequency
    fe = (num_poles / 2.0) * mechanical_speed / (2.0 * np.pi) # rad/s to Hz

    # build B-fields for each element in tooth pair for each rotor position in [0, 2*pi] electric radians
    Bx, By, delta = build_tooth_pair_Bfields(Bs, mesh, xi_geometry, xi_discrete)
    
    # get derivative of B-field w.r.t. electrical angle
    pwn = 0.1 # Percentage of the Nyquist frequency to keep
    B_and_dBdtheta_e = lambda B: compute_B_and_dBdtheta_e(B, pwn=pwn)
    v_B_and_dBdtheta_e = vmap(B_and_dBdtheta_e, in_axes=0)

    Bx_hat, dBxdtheta_e, Bx_max, ks = v_B_and_dBdtheta_e(Bx)
    By_hat, dBydtheta_e, By_max, _ = v_B_and_dBdtheta_e(By)

    ## compute hysteresis loss
    ph_x = compute_stator_hysteresis_loss_component(Bx_hat, dBxdtheta_e, k_h, fe, num_teeth, axial_length, delta)
    ph_y = compute_stator_hysteresis_loss_component(By_hat, dBydtheta_e, k_h, fe, num_teeth, axial_length, delta)
    ph_s = ph_x + ph_y

    ## compute eddy current loss
    pc_s = compute_stator_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_teeth, axial_length, delta)

    ## compute excess loss
    pe_s = compute_stator_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_teeth, axial_length, delta)
    
    return ph_s, pc_s, pe_s


def main():
    
    # conversion factor from meters to centimeters
    m2cm = 1.0e2
    # conversion factor from centimeters to inches
    cm2in = 1/2.54

    idx_sample = 76
    load_dir = 'data/'

    num_rotor_positions = 64 # 64
    deg2rad = jnp.pi/180.0
    vec_rotor_positions = jnp.linspace(0.0, deg2rad*15.0, num_rotor_positions, endpoint=False)

    # load meshes
    filename = 'mesh_pmsm_s' + str(idx_sample) + '.h5'
    file_path = os.path.join(load_dir, filename)
    with h5py.File(file_path, 'r') as f:

        vertices = f['stator_vertices'][()]
        triangles = f['stator_triangles'][()]
        segments = f['stator_segments'][()]
        vertex_markers = f['stator_vertex_markers'][()]
        vertices_mask = f['stator_vertices_mask'][()]

        triangles_mask = f['stator_triangles_mask'][()]
        segment_markers = f['stator_segment_markers'][()]
        segments_mask = f['stator_segments_mask'][()]

        stator_mesh = Mesh(vertices = jnp.array(vertices, dtype=jnp.float64),
                            vertex_markers = jnp.array(vertex_markers, dtype=jnp.int32),
                            vertices_mask = jnp.array(vertices_mask, dtype=jnp.bool_),
                            triangles = jnp.array(triangles, dtype=jnp.int32),
                            triangles_mask = jnp.array(triangles_mask, dtype=jnp.bool_),
                            segments = jnp.array(segments, dtype=jnp.int32),
                            segment_markers = jnp.array(segment_markers, dtype=jnp.int32),
                            segments_mask = jnp.array(segments_mask, dtype=jnp.bool_))
        
        # rotor
        vertices = f['rotor_vertices'][()]
        triangles = f['rotor_triangles'][()]
        segments = f['rotor_segments'][()]
        vertex_markers = f['rotor_vertex_markers'][()]
        vertices_mask = f['rotor_vertices_mask'][()]

        triangles_mask = f['rotor_triangles_mask'][()]
        segment_markers = f['rotor_segment_markers'][()]
        segments_mask = f['rotor_segments_mask'][()]

        _rotor_mesh = Mesh(vertices = jnp.array(vertices, dtype=jnp.float64),
                            vertex_markers = jnp.array(vertex_markers, dtype=jnp.int32),
                            vertices_mask = jnp.array(vertices_mask, dtype=jnp.bool_),
                            triangles = jnp.array(triangles, dtype=jnp.int32),
                            triangles_mask = jnp.array(triangles_mask, dtype=jnp.bool_),
                            segments = jnp.array(segments, dtype=jnp.int32),
                            segment_markers = jnp.array(segment_markers, dtype=jnp.int32),
                            segments_mask = jnp.array(segments_mask, dtype=jnp.bool_))
        
        # air gap
        vertices = f['air_gap_vertices'][()]
        triangles = f['air_gap_triangles'][()]
        segments = f['air_gap_segments'][()]

        vertex_markers = f['air_gap_vertex_markers'][()]
        vertices_mask = f['air_gap_vertices_mask'][()]

        triangles_mask = f['air_gap_triangles_mask'][()]
        segment_markers = f['air_gap_segment_markers'][()]
        segments_mask = f['air_gap_segments_mask'][()]

        air_gap_mesh = Mesh(vertices = jnp.array(vertices, dtype=jnp.float64),
                            vertex_markers = jnp.array(vertex_markers, dtype=jnp.int32),
                            vertices_mask = jnp.array(vertices_mask, dtype=jnp.bool_),
                            triangles = jnp.array(triangles, dtype=jnp.int32),
                            triangles_mask = jnp.array(triangles_mask, dtype=jnp.bool_),
                            segments = jnp.array(segments, dtype=jnp.int32),
                            segment_markers = jnp.array(segment_markers, dtype=jnp.int32),
                            segments_mask = jnp.array(segments_mask, dtype=jnp.bool_))  

    # load fea data
    filename = 'pmsm_s' + str(idx_sample) + '_J_peak5_fem.h5'
    file_path = os.path.join(load_dir, filename)
    with h5py.File(file_path, 'r') as f:
    
        As_fea = f['As'][0]
        Ar_fea = f['Ar'][0]
        Aag_fea = f['Aag'][0]
        Bs_fea = f['Bs'][0]
        Br_fea = f['Br'][0]
        Br_rrf_fea = f['Br_rrf'][0]
        Bag_fea = f['Bag'][0]
    

    # B magnitude
    #Bs_fea_mag = np.linalg.norm(Bs_fea, axis=1)
    #Br_fea_mag = np.linalg.norm(Br_fea, axis=1)
    #Bag_fea_mag = np.linalg.norm(Bag_fea, axis=1)

    #A_fem = jnp.hstack([As_fea, Ar_fea, Aag_fea])
    
    #B_fem = jnp.hstack([Bs_fea_mag, Br_fea_mag, Bag_fea_mag])

    # rotor_mesh = _rotate_mesh(_rotor_mesh, rotor_position)

    # merge the meshes
    # vertices = jnp.vstack([stator_mesh.vertices, rotor_mesh.vertices, air_gap_mesh.vertices])
    # triangles = jnp.vstack([stator_mesh.triangles,
    #                        rotor_mesh.triangles + stator_mesh.vertices.shape[0],
    #                        air_gap_mesh.triangles + stator_mesh.vertices.shape[0] + rotor_mesh.vertices.shape[0]])

    # Create triangulation for ploting
    #x = m2cm*vertices[:, 0]
    #y = m2cm*vertices[:, 1]
    #trian = triangles
    # triangulation = Triangulation(x, y, trian)


if __name__ == "__main__":
    main()
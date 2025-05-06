import os
import h5py
import numpy as np

def build_tooth_pair_Bfields(Br_s, Bt_s):
    # Number of triangles in tooth pair core
    n_tri_tpc = Br_s.shape[1] // 3

    # Reconstruct B-fields for each element in a tooth pair
    Bt_tp = np.vstack((
        Bt_s[:, 2*n_tri_tpc:], 
        Bt_s[:, n_tri_tpc:2*n_tri_tpc], 
        Bt_s[:, :n_tri_tpc],
       -Bt_s[:, 2*n_tri_tpc:],
       -Bt_s[:, n_tri_tpc:2*n_tri_tpc],
       -Bt_s[:, :n_tri_tpc]
    ))
    
    Br_tp = np.vstack((
        Br_s[:, 2*n_tri_tpc:],
        Br_s[:, n_tri_tpc:2*n_tri_tpc],
        Br_s[:, :n_tri_tpc],
       -Br_s[:, 2*n_tri_tpc:],
       -Br_s[:, n_tri_tpc:2*n_tri_tpc],
       -Br_s[:, :n_tri_tpc]
    ))

    return Bt_tp.T, Br_tp.T

def build_rotor_pole_Bfields(Br_r_rrf, Bt_r_rrf):
    # Reconstruct B-fields for each element in a rotor pole
    Br = np.tile(Br_r_rrf, (6, 1))
    Bt = np.tile(Bt_r_rrf, (6, 1))

    return Br.T, Bt.T

def compute_B_and_dBdtheta_e(B, pwn):
    # Define domain and compute FFT of B
    L = 2.0 * np.pi
    N = len(B)
    B_fft = np.fft.rfft(B)
    
    # Filter high frequency components
    msk = np.ones(N // 2 + 1)
    msk[int(pwn * N //2):] = 0
    B_hat = np.fft.irfft(msk * B_fft, n=N)
    
    # Compute derivative using FFT frequencies
    k = np.fft.rfftfreq(N, d=L / (N * 2 * np.pi))
    k[int(pwn * N//2):] = 0.0
    dBdtheta_e = np.fft.irfft(1.0j * k * B_fft, n=N)
    
    # Compute Fourier series amplitude (account for negative frequencies)
    C = np.abs(B_fft) / N
    Bmax = 2.0 * C * msk

    return B_hat, dBdtheta_e, Bmax, k

def process_field_data(B_array, pwn):
    # Process B-field data to compute B_hat, dBdtheta_e, Bmax, and ks
    results = [compute_B_and_dBdtheta_e(B, pwn) for B in B_array]
    B_hat = np.array([r[0] for r in results])
    dBtheta_e = np.array([r[1] for r in results])
    B_max = np.array([r[2] for r in results])
    ks = np.array([r[3] for r in results])

    return B_hat, dBtheta_e, B_max, ks

def process_two_field_data(B1, B2, pwn):
    # Process two B-field data arrays to compute B_hat, dBdtheta_e, Bmax, and ks
    B1_hat, dB1_dtheta, B1_max, ks = process_field_data(B1, pwn)
    B2_hat, dB2_dtheta, B2_max, _ = process_field_data(B2, pwn)

    return B1_hat, dB1_dtheta, B1_max, ks, B2_hat, dB2_dtheta, B2_max

def zero_crossings(signal):
    # Find zero crossings in the signal
    signal = np.array(signal)
    upper_threshold = np.max(np.abs(signal)) * 0.05 
    outputs = [1.0]
    output = -1.0

    for val in signal[1:]:
        if val > upper_threshold:
            output = 1.0
        elif output == 1.0 and val < 0.0:
            output = -1.0
        outputs.append(output)

    crossings = np.where(np.diff(np.sign(np.array(outputs))))[0]
    return crossings

def _compute_bmax_bmin_common(B, crossings):
    # Compute bmax and bmin from B and crossings
    B_crossings = B[crossings]
    max_or_min = np.ones_like(B_crossings)
    for k in range(1, len(B_crossings)):
        max_or_min[k] = -max_or_min[k - 1]

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

    return np.array(bmaxl), np.array(bminl), np.array(crossmaxl), np.array(crossminl)

def compute_bmax_bmin_stator(B, dBdt):

    idx_bmax = np.argmax(B)
    B = np.roll(B, -idx_bmax)
    dBdt = np.roll(dBdt, -idx_bmax)
    crossings_first_half = zero_crossings(dBdt[: len(dBdt) // 2])
    if (len(crossings_first_half) - 1) % 2 != 0:
        crossings_first_half = crossings_first_half[:-1]
    crossings_second_half = crossings_first_half + len(dBdt) // 2
    crossings = np.concatenate([crossings_first_half, crossings_second_half])

    return _compute_bmax_bmin_common(B, crossings)

def compute_bmax_bmin_rotor(B, dBdt):

    idx_bmax = np.argmax(B)
    B = np.roll(B, -idx_bmax)
    dBdt = np.roll(dBdt, -idx_bmax)
    crossings = zero_crossings(dBdt)
    if len(crossings) % 2 != 0:
        crossings = crossings[:-1]

    return _compute_bmax_bmin_common(B, crossings)

def compute_hysteresis_loss_component_generic(B, dBdtheta_e, k_h, fe, axial_length, delta,
                                              scale, bmax_bmin_func, slice_length=None):

    ph_bar = np.zeros(B.shape[0])
    if slice_length is not None:
        B_proc = B[:, :slice_length]
        dBdtheta_proc = dBdtheta_e[:, :slice_length]
    else:
        B_proc = B
        dBdtheta_proc = dBdtheta_e

    for i in range(B.shape[0]):
        bmax, bmin, _, _ = bmax_bmin_func(B_proc[i], dBdtheta_proc[i])
        ellipse_bm = (bmax - bmin) / 2.0
        ellipse_areas = k_h * ellipse_bm**2
        ph_bar[i] =  np.sum(ellipse_areas)

    return fe * axial_length * scale * np.sum(ph_bar * delta)

def compute_eddy_current_loss_generic(ks, Bx_max, By_max, k_c, fe, axial_length, delta, scale):
 
    pc_x_bar = np.sum(ks**2 * Bx_max**2, axis=1)
    pc_y_bar = np.sum(ks**2 * By_max**2, axis=1)
    pc_bar = pc_x_bar + pc_y_bar

    return axial_length * scale * k_c * (fe**2) * np.sum(pc_bar * delta)

def compute_excess_loss_generic(dBxdtheta_e, dBydtheta_e, k_e, fe, axial_length, delta, scale):
  
    Ce = 8.763363 
    ke_prime = k_e / Ce
    we = 2 * np.pi * fe
    pe_bar = np.mean((dBxdtheta_e**2 + dBydtheta_e**2)**0.75, axis=1)

    return axial_length * scale * ke_prime * (we**1.5) * np.sum(pe_bar * delta)

def compute_stator_hysteresis_loss(Bx, dBxdtheta_e, By, dBydtheta_e, k_h, fe, 
                                   num_teeth, axial_length, delta):
   
    ph_x = compute_hysteresis_loss_component_generic(
        Bx, dBxdtheta_e, k_h, fe, axial_length, delta,
        scale=(num_teeth / 2),
        bmax_bmin_func=compute_bmax_bmin_stator
    )

    ph_y = compute_hysteresis_loss_component_generic(
        By, dBydtheta_e, k_h, fe, axial_length, delta,
        scale=(num_teeth / 2),
        bmax_bmin_func=compute_bmax_bmin_stator
    )

    return ph_x + ph_y

def compute_rotor_hysteresis_loss(Bx, dBxdtheta_e, By, dBydtheta_e, k_h, fe, 
                                  num_poles, num_teeth_per_pole, axial_length, delta):

    slice_length = Bx.shape[1] // int(num_teeth_per_pole)
    ph_x = compute_hysteresis_loss_component_generic(
        Bx, dBxdtheta_e, k_h, fe, axial_length, delta,
        scale=num_poles*num_teeth_per_pole,
        bmax_bmin_func=compute_bmax_bmin_rotor,
        slice_length=slice_length
    )

    ph_y = compute_hysteresis_loss_component_generic(
        By, dBydtheta_e, k_h, fe, axial_length, delta,
        scale=num_poles*num_teeth_per_pole,
        bmax_bmin_func=compute_bmax_bmin_rotor,
        slice_length=slice_length
    )

    return ph_x + ph_y

def compute_stator_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_teeth, axial_length, delta):
    return compute_eddy_current_loss_generic(ks, Bx_max, By_max, k_c, fe, axial_length, delta,
                                             scale=(num_teeth / 2))

def compute_rotor_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_poles, axial_length, delta):
    return compute_eddy_current_loss_generic(ks, Bx_max, By_max, k_c, fe, axial_length, delta,
                                             scale=num_poles)

def compute_stator_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_teeth, axial_length, delta):
    return compute_excess_loss_generic(dBxdtheta_e, dBydtheta_e, k_e, fe, axial_length, delta,
                                       scale=(num_teeth / 2))

def compute_rotor_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_poles, axial_length, delta):
    return compute_excess_loss_generic(dBxdtheta_e, dBydtheta_e, k_e, fe, axial_length, delta,
                                       scale=num_poles)

def get_machine_constants():

    mm = 1.0e-3
    rpm2rad_per_sec = 2.0 * np.pi / 60.0
    axial_length = 90.0 * mm
    mechanical_speed = rpm2rad_per_sec * 1000.0
    num_poles = 8
    num_teeth = 48
    k_h = 176.0
    k_c = 0.102
    k_e = 2.72

    return axial_length, mechanical_speed, num_poles, num_teeth, k_h, k_c, k_e

def compute_stator_core_loss(Br_s, Bt_s, delta):
    axial_length, mechanical_speed, num_poles, num_teeth, k_h, k_c, k_e = get_machine_constants()
    fe = (num_poles / 2.0) * mechanical_speed / (2.0 * np.pi)

    # Build B-fields for each element in the tooth pair for all rotor positions.
    Bx, By = build_tooth_pair_Bfields(Br_s, Bt_s)
    pwn = 0.1  # Percentage of the Nyquist frequency to keep (\alpha).

    # Process both Bx and By.
    Bx_hat, dBxdtheta_e, Bx_max, ks, By_hat, dBydtheta_e, By_max = process_two_field_data(Bx, By, pwn)

    # Compute total stator hysteresis loss from both components.
    ph_s = compute_stator_hysteresis_loss(
        Bx_hat, dBxdtheta_e, By_hat, dBydtheta_e,
        k_h, fe, num_teeth, axial_length, delta
    )
    pc_s = compute_stator_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_teeth, axial_length, delta)
    pe_s = compute_stator_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_teeth, axial_length, delta)
    
    return ph_s, pc_s, pe_s

def compute_rotor_core_loss(Br_r_rrf, Bt_r_rrf, delta):
    axial_length, mechanical_speed, num_poles, num_teeth, k_h, k_c, k_e = get_machine_constants()
    num_teeth_per_pole = num_teeth / num_poles
    fe = (num_poles / 2.0) * mechanical_speed / (2.0 * np.pi)

    # Build B-fields for one rotor pole for all rotor positions.
    Bx, By = build_rotor_pole_Bfields(Br_r_rrf, Bt_r_rrf)
    pwn = 0.1  # Percentage of the Nyquist frequency to keep (\alpha).

    # Process both Bx and By together.
    Bx_hat, dBxdtheta_e, Bx_max, ks, By_hat, dBydtheta_e, By_max = process_two_field_data(Bx, By, pwn)

    ph_r = compute_rotor_hysteresis_loss(
        Bx_hat, dBxdtheta_e, By_hat, dBydtheta_e,
        k_h, fe, num_poles, num_teeth_per_pole, axial_length, delta
    )
    pc_r = compute_rotor_eddy_current_loss(ks, Bx_max, By_max, k_c, fe, num_poles, axial_length, delta)
    pe_r = compute_rotor_excess_loss(dBxdtheta_e, dBydtheta_e, k_e, fe, num_poles, axial_length, delta)
    
    return ph_r, pc_r, pe_r

def main():

    # load data from HDF5 file
    filename = 'fea-data.h5'
    current_file = os.path.realpath(__file__)
    current_directory = os.path.dirname(current_file)
    file_path = os.path.join(current_directory, filename)
    with h5py.File(file_path, 'r') as f:
        Br_s = f['Br_s'][()]
        Bt_s = f['Bt_s'][()]
        delta_s = f['delta_s'][()]
        Br_r_rrf = f['Br_r_rrf'][()]
        Bt_r_rrf = f['Bt_r_rrf'][()]
        delta_r = f['delta_r'][()]

    # Compute and display stator core losses
    ph_s, pc_s, pe_s = compute_stator_core_loss(Br_s, Bt_s, delta_s)
    print("\nStator hysteresis loss: ", ph_s, " W")
    print("Stator eddy current loss: ", pc_s, " W")
    print("Stator excess loss: ", pe_s, " W")
    print("Total stator core loss: ", ph_s + pc_s + pe_s, " W")

    # Compute and display rotor core losses
    ph_r, pc_r, pe_r = compute_rotor_core_loss(Br_r_rrf, Bt_r_rrf, delta_r)
    print("\nRotor hysteresis loss: ", ph_r, " W")
    print("Rotor eddy current loss: ", pc_r, " W")
    print("Rotor excess loss: ", pe_r, " W")
    print("Total rotor core loss: ", ph_r + pc_r + pe_r, " W")

if __name__ == "__main__":
    main()

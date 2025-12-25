from __future__ import division
"""
test_bs_ris_channel_est.py: Tests for VAMP channel estimation in BS-RIS systems

This module tests EM-VAMP for baseline and RIS channel estimation.
System model: y = A * h + w, where h is sparse in some domain
"""

# Add the path to the vampyre package and import it
# Add the path to the vampyre package
import os
import sys
for d in ('..', os.path.join('..', '..')):
    fd = os.path.abspath(os.path.join(os.path.dirname(__file__), d))
    if fd not in sys.path:
        sys.path.append(fd)

import vampyre as vp
import numpy as np
import unittest

def bs_ris_channel_vamp(nt=8, nr=4, nris=16, nsubcarr=32, sparse_ratio=0.2,
                        snr=20, verbose=False, nit=30, tune=True,
                        vamp_meth='vamp', plot_res=False):
    """
    Test EM-VAMP for BS-RIS channel estimation.

    System Model: y = A @ h + w
    where A is sensing matrix, h is sparse channel, w is AWGN.

    :param int nt: TX antennas
    :param int nr: RX antennas
    :param int nris: RIS elements
    :param int nsubcarr: Subcarriers
    :param float sparse_ratio: Sparsity ratio
    :param float snr: SNR in dB
    :param bool verbose: Print results
    :param int nit: Iterations
    :param bool tune: Enable tuning
    :param str vamp_meth: 'vamp' or 'mlvamp'
    :param bool plot_res: Plot results
    """

    # Parameters
    map_est = False
    is_complex = True

    # Dimensions
    nh = nt * nris
    ny = nr * nsubcarr
    hshape = (nh, nsubcarr)
    yshape = (ny, nsubcarr)

    # GMM for sparse channel
    varc_lo = 1e-4
    varc_hi = 1.0
    prob_hi = sparse_ratio
    meanc = np.array([0, 0])
    probc = np.array([1 - prob_hi, prob_hi])
    varc = np.array([varc_lo, varc_hi])
    nc = len(probc)

    # Generate sparse channel
    hlen = np.prod(hshape)
    ind = np.random.choice(nc, hlen, p=probc)
    u = np.random.randn(hlen) + 1j * np.random.randn(hlen)
    h0_real = u.real * np.sqrt(varc[ind] / 2) + meanc[ind]
    h0_imag = u.imag * np.sqrt(varc[ind] / 2) + meanc[ind]
    h0 = (h0_real + 1j * h0_imag).reshape(hshape)

    # Sensing matrix
    A_real = np.random.randn(ny, nh) / np.sqrt(nh)
    A_imag = np.random.randn(ny, nh) / np.sqrt(nh)
    A = A_real + 1j * A_imag

    # Received signal
    y0 = A.dot(h0)
    wvar = np.power(10, -0.1 * snr) * np.mean(np.abs(y0) ** 2)
    w_real = np.random.normal(0, np.sqrt(wvar / 2), yshape)
    w_imag = np.random.normal(0, np.sqrt(wvar / 2), yshape)
    y = y0 + w_real + 1j * w_imag

    # Message handlers
    msg_hdl0 = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,
                                   shape=hshape)
    if vamp_meth == 'mlvamp':
        msg_hdl1 = vp.estim.MsgHdlSimp(map_est=map_est, is_complex=is_complex,
                                       shape=yshape)
        msg_hdl_list = [msg_hdl0, msg_hdl1]

    # Tuning initialization
    yvar = np.mean(np.abs(y) ** 2)
    meanc_init = np.array([0, 0])
    prob1 = np.minimum(ny / nh / 2, 0.95)
    var1 = yvar / np.mean(np.abs(A) ** 2) / nh / prob1
    probc_init = np.array([1 - prob1, prob1])
    varc_init = np.array([1e-4, var1])
    mean_fix = [1, 0]
    var_fix = [1, 0]

    # Linear estimator
    Aop = vp.trans.MatrixLT(A, hshape)
    b = np.zeros(yshape)

    if vamp_meth == 'mlvamp':
        est_lin = vp.estim.LinEstTwo(Aop, b, map_est=map_est)
    elif tune:
        est_lin = vp.estim.LinEst(Aop, y, wvar=yvar, map_est=map_est,
                                  tune_wvar=True)
    else:
        est_lin = vp.estim.LinEst(Aop, y, wvar=wvar, map_est=map_est,
                                  tune_wvar=False)

    # Channel input estimator
    if tune:
        est_in = vp.estim.GMMEst(shape=hshape, zvarmin=1e-6, tune_gmm=True,
                                 probc=probc_init, meanc=meanc_init,
                                 varc=varc_init, mean_fix=mean_fix,
                                 var_fix=var_fix)
    else:
        est_in = vp.estim.GMMEst(shape=hshape, probc=probc, meanc=meanc,
                                 varc=varc, tune_gmm=False)

    # ML-VAMP output estimator
    if vamp_meth == 'mlvamp':
        if tune:
            est_out = vp.estim.GaussEst(y, zvar=yvar, shape=yshape,
                                        zmean_axes=[], tune_zvar=True)
        else:
            est_out = vp.estim.GaussEst(y, zvar=wvar, shape=yshape,
                                        zmean_axes=[], tune_zvar=False)
        est_list = [est_in, est_lin, est_out]

    # Create solver
    if vamp_meth == 'mlvamp':
        solver = vp.solver.MLVamp(est_list, msg_hdl_list,
                                 hist_list=['zhat', 'zhatvar'],
                                 comp_cost=True, nit=nit)
    else:
        solver = vp.solver.Vamp(est_in, est_lin, msg_hdl0,
                               hist_list=['zhat', 'zhatvar'],
                               comp_cost=True, nit=nit)

    # Run solver
    solver.solve()

    # Compute MSE
    zhat_hist = solver.hist_dict['zhat']
    zhatvar_hist = solver.hist_dict['zhatvar']
    nit2 = len(zhat_hist)

    hpow = np.mean(np.abs(h0) ** 2)
    mse_act = np.zeros(nit2)
    mse_pred = np.zeros(nit2)

    for it in range(nit2):
        if vamp_meth == 'mlvamp':
            zhati = zhat_hist[it][0]
            zhatvari = zhatvar_hist[it][0]
        else:
            zhati = zhat_hist[it]
            zhatvari = zhatvar_hist[it]

        herr = np.mean(np.abs(zhati - h0) ** 2)
        hhatvar = np.mean(zhatvari)
        mse_act[it] = 10 * np.log10(herr / hpow)
        mse_pred[it] = 10 * np.log10(hhatvar / hpow)

    # Final MSE
    mse_tol = -10
    fail = (mse_act[-1] > mse_tol)

    if fail or verbose:
        print('Channel Est MSE {0:s}: act {1:7.2f} dB, pred: {2:7.2f} dB'.format(
            vamp_meth, mse_act[-1], mse_pred[-1]))

    if fail:
        raise vp.common.TestException('MSE exceeded expected value')

    return mse_act[-1], mse_pred[-1]


class TestCases(unittest.TestCase):
    def test_bs_ris_channel_vamp(self):
        """
        Test EM-VAMP for BS-RIS channel estimation
        """
        verbose = True
        plot_res = False

        # Test with smaller antenna size and single SNR
        print('\n=== Testing BS-RIS Channel Estimation with VAMP ===')
        mse_vamp = bs_ris_channel_vamp(
            nt=4, nr=2, nris=8, nsubcarr=16,
            sparse_ratio=0.2, snr=20,
            vamp_meth='vamp', tune=True,
            verbose=verbose, plot_res=plot_res)

        # Test with ML-VAMP
        print('\n=== Testing BS-RIS Channel Estimation with ML-VAMP ===')
        mse_mlvamp = bs_ris_channel_vamp(
            nt=4, nr=2, nris=8, nsubcarr=16,
            sparse_ratio=0.2, snr=20,
            vamp_meth='mlvamp', tune=True,
            verbose=verbose, plot_res=plot_res)

        print('\n=== Test Results ===')
        print('VAMP final MSE: {0:.2f} dB'.format(mse_vamp[0]))
        print('ML-VAMP final MSE: {0:.2f} dB'.format(mse_mlvamp[0]))


if __name__ == '__main__':
    unittest.main()

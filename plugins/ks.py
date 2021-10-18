#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
from pullvals import pull
import root_numpy
import numpy as np
from pullvals import pull, maxPullNorm
import scipy
import scipy.stats

def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=100000, **kwargs):


    data_name = histpair.data_name
    ref_name = histpair.ref_name


    data_hist = histpair.data_hist
    ref_hist_list = [x.values for x in histpair.ref_hists_list]
    
    
    
    #ref_hists_list = [x.values for x in histpair.ref_hists_list]

    # check for 1d hists and that refs are not empty
    if "1" not in str(type(data_hist)) :
        return None
    for i in ref_hist_list:
        if i.sum() == 0:
            return None
    if data_hist.values.sum() == 0: 
        return None


    data_hist_norm = np.copy(data_hist.values)
    data_hist_Entries = np.sum(data_hist_norm)
    ref_hist_Entries = np.mean(np.sum(ref_hist_list))


    ## calculate the ref arrays 
    ref_hist_arr = np.array(ref_hist_list)
    
    ref_hist_arr = np.array([x*data_hist_norm.sum()/x.sum() if x.sum() > 0 else x for x in ref_hist_list])
    ref_hist_norm = ref_hist_arr.mean(axis=0)
    ref_hist_errs = ref_hist_arr.std(axis=0)


    
    ## data arrays
    '''data_hist_scale = ref_hist_norm.sum()/data_hist_norm.sum()
    data_hist_norm*=data_hist_scale'''
    # data_hist_errs = np.sqrt(data_hist_norm*data_hist_scale)
    data_hist_errs = np.nan_to_num(abs(np.array(scipy.stats.chi2.interval(0.6827, 2 * data_hist_norm)) / 2 - 1 - data_hist_norm))

    # import pickle
    # pickle.dump(data_hist_errs, open(f'pickles/dataErr-{data_name}-sqrt.pkl','wb'))
    # raise(ValueError)
    
    

    # ks = data_hist.KolmogorovTest(ref_hist, "M")
    ks = scipy.stats.kstest(ref_hist_norm, data_hist_norm)[0]

    is_good = data_hist_Entries > 0
    is_outlier = is_good and ks > ks_cut

    # canv, artifacts = None, # draw_same(
        #data_hist_norm, histpair.data_run, ref_hist_norm, histpair.ref_run)
    

    pull_cap = 25
    ## chi2 and pull vals
    max_pull = 0
    nBins = 0
    chi2 = 0
    
    nBinsUsed = np.count_nonzero(np.add(ref_hist_norm, data_hist_norm))
    
    pulls = np.zeros_like(ref_hist_norm)
    #for i in range(1, data_hist.GetNbinsX() + 1):
    for i in range(0, data_hist_norm.shape[0]):
        # Bin 1 data
        bin1 = data_hist_norm[i]

        # Bin 2 data
        bin2 = ref_hist_norm[i]#ref_hist.GetBinContent(i)

        # Getting Proper Poisson error 
        '''bin1err, bin2err = data_hist_errs[i], ref_hist_errs[i]'''
        bin1err, bin2err = data_hist_errs[0, i], ref_hist_errs[i]
        if bin1 < bin2:
            bin1err, bin2err = data_hist_errs[1, i], ref_hist_errs[i]

        # Count bins for chi2 calculation
        nBins += 1

        # Ensure that divide-by-zero error is not thrown when calculating pull
        if bin1err == 0 and bin2err == 0:
            new_pull = 0
        else:
            new_pull = pull(bin1, bin1err, bin2, bin2err)
            #new_pull = maxPullNorm(new_pull, nBinsUsed)

        # Sum pulls
        chi2 += new_pull**2

        # Check if max_pull
        #max_pull = max(max_pull, abs(new_pull))
        #max_pull = maxPullNorm(max_pull, nBinsUsed)


        # Clamp the displayed value
        # fill_val = max(min(new_pull, pull_cap), -pull_cap)
        #fill_val = max_pull
    
        # If the input bins were explicitly empty, make this bin white by
        # setting it out of range
        ## why is this done????
        #if bin1 == bin2 == 0:
        #    fill_val = -999
        
        pulls[i] = new_pull
        
    # Compute chi2
    if nBinsUsed > 0:
        chi2 = (chi2 / nBinsUsed)
        pulls = maxPullNorm(pulls, nBinsUsed)
        max_pull = pulls.max()
    else:
        chi2=0
        max_pull = 0 

    
    
    
    canv = None
    artifacts = [pulls]

    info = {
        'Data_Entries': data_hist_Entries,
        'Ref_Entries': ref_hist_Entries,
        'KS_Val': ks,
        'Chi_Squared' : chi2,
        'Max_Pull_Val': max_pull,
        'nBins' : nBins,
        'pulls' : (pulls, data_hist.edges)
    }

    return PluginResults(
        canv,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


# def pull(bin1, binerr1, bin2, binerr2):
#     ''' Calculate the pull value between two bins.
#         pull = (data - expected)/sqrt(sum of errors in quadrature))
#         data = |bin1 - bin2|, expected = 0
#     '''
#     ## changing to pull with tolerance
#     # return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)
#     return np.abs(bin1 - bin2)/(np.sqrt(np.power(binerr1,2)+np.power(binerr2,2)+0.01*(bin1+bin2)))

# def maxPullNorm(maxPull, nBinsUsed):
#     prob = ROOT.TMath.Prob(np.power(maxPull, 2),1)
#     probNorm = 1-np.power((1-prob),nBinsUsed)
#     ## .9999999999999999 is the max that can go into chi2quantile
#     val = (1-probNorm) 
#     val = val if val < .9999999999999999 else .9999999999999999
#     return np.sqrt(ROOT.TMath.ChisquareQuantile(val,1))

def draw_same(data_hist, data_run, ref_hist, ref_run):
    # Set up canvas
    c = ROOT.TCanvas('c', 'c')
    data_hist = data_hist.Clone()
    ref_hist = ref_hist.Clone()

    # Ensure plot accounts for maximum value
    ref_hist.SetMaximum(
        max(data_hist.GetMaximum(), ref_hist.GetMaximum()) * 1.1)

    ROOT.gStyle.SetOptStat(1)
    ref_hist.SetStats(True)
    data_hist.SetStats(True)

    # Set hist style
    ref_hist.SetLineColor(28)
    ref_hist.SetFillColor(20)
    ref_hist.SetLineWidth(1)
    data_hist.SetLineColor(ROOT.kRed)
    data_hist.SetLineWidth(1)

    # Name histograms
    ref_hist.SetName("Reference")
    data_hist.SetName("Data")

    # Plot hist
    ref_hist.Draw()
    data_hist.Draw("sames hist e")
    c.Update()

    # Modify stats boxes
    r_stats = ref_hist.FindObject("stats")
    f_stats = data_hist.FindObject("stats")

    r_stats.SetY1NDC(0.15)
    r_stats.SetY2NDC(0.30)
    r_stats.SetTextColor(28)
    r_stats.Draw()

    f_stats.SetY1NDC(0.35)
    f_stats.SetY2NDC(0.50)
    f_stats.SetTextColor(ROOT.kRed)
    f_stats.Draw()

    # Text box
    data_text = ROOT.TLatex(.52, .91, "#scale[0.6]{Data: " + str(data_run) + "}")
    ref_text = ROOT.TLatex(.72, .91, "#scale[0.6]{Ref: " + str(ref_run) + "}")
    data_text.SetNDC(ROOT.kTRUE)
    ref_text.SetNDC(ROOT.kTRUE)
    data_text.Draw()
    ref_text.Draw()

    c.Update()
    artifacts = [data_hist, data_text] 
    #artifacts = [data_hist, ref_hist, data_text, ref_text]
    return c, artifacts



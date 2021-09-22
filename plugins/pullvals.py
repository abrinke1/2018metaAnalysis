#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
import numpy as np
import time
import root_numpy

def comparators():
    return {
        'pull_values': pullvals
    }


def pullvals(histpair,
             pull_cap=25, chi2_cut=500, pull_cut=20, min_entries=100000, norm_type='all',
             **kwargs):
    """Can handle poisson driven TH2s or generic TProfile2Ds"""
    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist
    ref_hists_list = histpair.ref_hists_list
    
    data_histD = ROOT.TH2D()
    data_hist.Copy(data_histD)
    data_hist = data_histD
    ref_histD = ROOT.TH2D()
    ref_hist.Copy(ref_histD)
    ref_hist = ref_histD

    # Check that the hists are histograms
    if not data_hist.InheritsFrom('TH1') or not ref_hist.InheritsFrom('TH1'):
        return None

    # Check that the hists are 2 dimensional
    if data_hist.GetDimension() != 2 or ref_hist.GetDimension() != 2:
        return None

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPalette(ROOT.kLightTemperature)
    ROOT.gStyle.SetNumberContours(255)

    # Get empty clone of reference histogram for pull hist
    if data_hist.InheritsFrom('TProfile2D'):
        pull_hist = ref_hist.ProjectionXY("pull_hist")
    else:
        pull_hist = ref_hist.Clone("pull_hist")
    pull_hist.Reset()

    # Reject empty histograms
    is_good = data_hist.GetEntries() != 0 and data_hist.GetEntries() >= min_entries

    # Normalize data_hist
    # if norm_type == "row":
    #     normalize_rows(data_hist, ref_hist)
    # else:
    #     if data_hist.GetEntries() > 0:
    #         data_hist.Scale(ref_hist.GetSumOfWeights() / data_hist.GetSumOfWeights())
 
    histscale = 1#ref_hist.GetSumOfWeights()#1
    if data_hist.GetEntries() > 0: 
        data_hist.Scale(histscale/data_hist.GetSumOfWeights())
    # if ref_hist.GetEntries() > 0: 
    #     ref_hist.Scale(histscale/ref_hist.GetSumOfWeights())
    for i in ref_hists_list:
        if i.GetEntries() > 0:
            i.Scale(histscale/i.GetSumOfWeights())
            
    # calculate the average of all ref_hists_list 
    avg_ref_hists_list = []
    for i in ref_hists_list:
        avg_ref_hists_list.append(root_numpy.hist2array(i))
    ref_hists_arr = np.array(avg_ref_hists_list)
    ref_hist = np.mean(ref_hists_arr, axis=0)
    ref_hist = ref_hist*1/ref_hist.sum()
    refErr = np.std(ref_hists_arr, axis=0)    


    max_pull = 0
    nBins = 0
    chi2 = 0
    
    ## caluclate nBinsUsed 
    data_arr = root_numpy.hist2array(data_hist)
    #ref_arr = root_numpy.hist2array(ref_hist)
    nBinsUsed = np.count_nonzero(np.add(ref_hist, data_arr))
    
    
    pulls = np.zeros_like(refErr)
    
    ## loop through bins to calculate max pull
    for x in range(1, data_hist.GetNbinsX() + 1):
        for y in range(1, data_hist.GetNbinsY() + 1):

            # Bin 1 data
            bin1 = data_hist.GetBinContent(x, y)

            # Bin 2 data
            bin2 = ref_hist[x-1,y-1]#ref_hist.GetBinContent(x, y)

            
            if not (bin1 + bin2 > 0):
                pulls[x-1,y-1] = 0
                continue
            
            # TEMPERARY - Getting Symmetric Error - Need to update with >Proper Poisson error 
            if data_hist.InheritsFrom('TProfile2D'):
                bin1err = data_hist.GetBinError(x, y)
                # bin2err = ref_hist.GetBinError(x, y)
                bin2err = refErr[x-1,y-1] # -1 because root index from 1 apparently
            else:
                # bin1err, bin2err = bin1**(.5), bin2**(.5)
                bin1err, bin2err = bin1**(.5), refErr[x-1, y-1]
            # Count bins for chi2 calculation
            nBins += 1 

            
            # Ensure that divide-by-zero error is not thrown when calculating pull
            if bin1err == 0 and bin2err == 0:
                new_pull = 0
            else:
                new_pull = pull(bin1, bin1err, bin2, bin2err)
                    
            pulls[x-1,y-1] = new_pull
            # Sum pulls
            chi2 += new_pull**2

            # Check if max_pull
            max_pull = max(max_pull, abs(new_pull))

            # Clamp the displayed value
            fill_val = max(min(new_pull, pull_cap), -pull_cap)

            # If the input bins were explicitly empty, make this bin white by
            # setting it out of range
            if bin1 == bin2 == 0:
                fill_val = -999

            # Fill Pull Histogram
            pull_hist.SetBinContent(x, y, fill_val)


    ## make normed chi2 and maxpull
    if nBinsUsed > 0:
        chi2 = chi2/nBinsUsed  
        max_pull = maxPullNorm(max_pull, nBinsUsed)
    else:
        chi2 = 0
        max_pull = 0
        
    
    is_outlier = is_good and (chi2 > chi2_cut or abs(max_pull) > pull_cut)

    # Set up canvas
    c = ROOT.TCanvas('c', 'Pull')

    # Plot pull hist
    pull_hist.GetZaxis().SetRangeUser(-(pull_cap), pull_cap)
    pull_hist.SetTitle(pull_hist.GetTitle() + " Pull Values")
    pull_hist.Draw("colz")

    # Text box
    data_text = ROOT.TLatex(.52, .91,
                            "#scale[0.6]{Data: " + str(histpair.data_run) + "}")
    ref_text = ROOT.TLatex(.72, .91,
                           "#scale[0.6]{Ref: " + str(histpair.ref_run) + "}")
    data_text.SetNDC(ROOT.kTRUE)
    ref_text.SetNDC(ROOT.kTRUE)
    data_text.Draw()
    ref_text.Draw()


    info = {
        'Chi_Squared': chi2,
        'Max_Pull_Val': max_pull,
        'Data_Entries': data_hist.GetEntries(),
        'Ref_Entries': ref_hist.sum(),
        'nBinsUsed' : nBinsUsed,
        'nBins' : nBins,
        'new_pulls' : pulls
    }

    artifacts = [pull_hist, data_text, ref_text]

    return PluginResults(
        c,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


def pull(bin1, bin1err, bin2, bin2err):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0
    '''
    ## changing to pull with tolerance
    # return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)
    return np.abs(bin1 - bin2)/(np.sqrt(np.power(bin1err,2)+np.power(bin2err,2))+0.01*(bin1+bin2))

def maxPullNorm(maxPull, nBinsUsed):
    ## `probGood = 1-scipy.stats.chi2.cdf(np.power(maxpull,2),1)` will give the same result as ROOT.TMath.Prob
    probGood = ROOT.TMath.Prob(np.power(maxPull, 2), 1)
    probBadNorm = np.power((1-probGood), nBinsUsed)
    val = min(probBadNorm, 1 - np.power(0.1,16))
    ## ChisquareQuantile can be substituted with `scipy.stats.chi2.ppf(val,1)`
    return np.sqrt(ROOT.TMath.ChisquareQuantile(val,1))
    

# def normalize_rows(data_hist, ref_hist):

#     for y in range(1, ref_hist.GetNbinsY() + 1):

#         # Stores sum of row elements
#         rrow = 0
#         frow = 0

#         # Sum over row elements
#         for x in range(1, ref_hist.GetNbinsX() + 1):

#             # Bin data
#             rbin = ref_hist.GetBinContent(x, y)
#             fbin = data_hist.GetBinContent(x, y)

#             rrow += rbin
#             frow += fbin

#         # Scaling factors
#         # Prevent divide-by-zero error
#         if frow == 0:
#             frow = 1
#         if frow > 0:
#             sf = float(rrow) / frow
#         else:
#             sf = 1
#         # Prevent scaling everything to zero
#         if sf == 0:
#             sf = 1

#         # Normalization
#         for x in range(1, data_hist.GetNbinsX() + 1):
#             # Bin data
#             fbin = data_hist.GetBinContent(x, y)
#             fbin_err = data_hist.GetBinError(x, y)

#             # Normalize bin
#             data_hist.SetBinContent(x, y, (fbin * sf))
#             data_hist.SetBinError(x, y, (fbin_err * sf))

#     return

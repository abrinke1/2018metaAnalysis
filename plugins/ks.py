#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ROOT
#from autodqm.plugin_results import PluginResults
from plugin_results import PluginResults
from pullvals import pull
import numpy as np


def comparators():
    return {
        "ks_test": ks
    }


def ks(histpair, ks_cut=0.09, min_entries=100000, **kwargs):

    data_name = histpair.data_name
    ref_name = histpair.ref_name

    data_hist = histpair.data_hist
    ref_hist = histpair.ref_hist


    # Check that the hists are histograms
    if not data_hist.InheritsFrom('TH1') or not ref_hist.InheritsFrom('TH1'):
        return None

    # Check that the hists are 1 dimensional
    if data_hist.GetDimension() != 1 or ref_hist.GetDimension() != 1:
        return None

    # Normalize data_hist
    if data_hist.GetEntries() > 0:
        data_hist.Scale(ref_hist.GetEntries() / data_hist.GetEntries())

    # Reject empty histograms
    is_good = data_hist.GetEntries() != 0 and data_hist.GetEntries() >= min_entries

    ks = data_hist.KolmogorovTest(ref_hist, "M")

    is_outlier = is_good and ks > ks_cut

    canv, artifacts = draw_same(
        data_hist, histpair.data_run, ref_hist, histpair.ref_run)


    pull_cap = 25
    ## chi2 and pull vals
    max_pull = 0
    nBins = 0
    nBinsUsed = 0
    chi2 = 0
    for i in range(1, ref_hist.GetNbinsX() + 1):
        # Bin 1 data
        bin1 = data_hist.GetBinContent(i)

        # Bin 2 data
        bin2 = ref_hist.GetBinContent(i)

        bin1err = data_hist.GetBinError(i)
        bin2err = ref_hist.GetBinError(i)

        # Count bins for chi2 calculation
        nBins += 1
        if (bin1+bin2) > 0: nBinsUsed +=1 

        # Ensure that divide-by-zero error is not thrown when calculating pull
        if bin1err == 0 and bin2err == 0:
            new_pull = 0
        else:
            new_pull = pull(bin1, bin1err, bin2, bin2err)
            new_pull = maxPullNorm(new_pull, nBinsUsed)

        # Sum pulls
        chi2 += new_pull**2

        # Check if max_pull
        max_pull = max(max_pull, abs(new_pull))
        #max_pull = maxPullNorm(max_pull, nBinsUsed)

        # Clamp the displayed value
        fill_val = max(min(new_pull, pull_cap), -pull_cap)

        # If the input bins were explicitly empty, make this bin white by
        # setting it out of range
        ## why is this done????
        if bin1 == bin2 == 0:
            fill_val = -999
    # Compute chi2
    chi2 = (chi2 / nBins)




    info = {
        'Data_Entries': data_hist.GetEntries(),
        'Ref_Entries': ref_hist.GetEntries(),
        'KS_Val': ks,
        'Chi_Squared' : chi2,
        'Max_Pull_Val': max_pull
    }

    return PluginResults(
        canv,
        show=is_outlier,
        info=info,
        artifacts=artifacts)


def pull(bin1, binerr1, bin2, binerr2):
    ''' Calculate the pull value between two bins.
        pull = (data - expected)/sqrt(sum of errors in quadrature))
        data = |bin1 - bin2|, expected = 0
    '''
    ## changing to pull with tolerance
    # return (bin1 - bin2) / ((binerr1**2 + binerr2**2)**0.5)
    return np.abs(bin1 - bin2)/(np.sqrt(np.power(binerr1,2)+np.power(binerr2,2)+0.01*(bin1+bin2)))

def maxPullNorm(maxPull, nBinsUsed):
    prob = ROOT.TMath.Prob(np.power(maxPull, 2),1)
    probNorm = 1-np.power((1-prob),nBinsUsed)
    ## .9999999999999999 is the max that can go into chi2quantile
    val = (1-probNorm) 
    val = val if val < .9999999999999999 else .9999999999999999
    return np.sqrt(ROOT.TMath.ChisquareQuantile(val,1))


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

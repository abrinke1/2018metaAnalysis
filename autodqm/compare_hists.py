#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import ROOT
#from autodqm import cfg
import cfg
import uproot
import numpy as np
#from autodqm.histpair import HistPair
from histpair import HistPair
# sys.path.insert(1, '/Users/si_sutantawibul1/Projects/AutoDQM/plugins')

def process(config_dir, subsystem,
            data_series, data_sample, data_run, data_path,
            ref_series, ref_sample, ref_run, ref_path,
            ref_list, ref_runs_list,
            output_dir='./out/', plugin_dir='./plugins/'):

    # Ensure no graphs are drawn to screen and no root messages are sent to
    # terminal
    ROOT.gROOT.SetBatch(ROOT.kTRUE)
    # Report only errors to stderr
    ROOT.gErrorIgnoreLevel = ROOT.kWarning + 1

    histpairs = compile_histpairs(config_dir, subsystem,
                                  data_series, data_sample, data_run, data_path,
                                  ref_series, ref_sample, ref_run, ref_path,
                                  ref_list, ref_runs_list)

    for d in [output_dir + s for s in ['/pdfs', '/jsons', '/pngs']]:
        if not os.path.exists(d):
            os.makedirs(d)

    hist_outputs = []

    comparator_funcs = load_comparators(plugin_dir)
    for hp in histpairs:
        try:
            comparators = [(c, comparator_funcs[c]) for c in hp.comparators]
        except KeyError as e:
            raise error("Comparator {} was not found.".format(str(e)))

        for comp_name, comparator in comparators:

            result_id = identifier(hp, comp_name)
            pdf_path = '{}/pdfs/{}.pdf'.format(output_dir, result_id)
            #json_path = '{}/jsons/{}.json'.format(output_dir, result_id)
            #png_path = '{}/pngs/{}.png'.format(output_dir, result_id)

            #if True:#not os.path.isfile(json_path):
            results = comparator(hp, **hp.config)

            # Continue if no results
            if not results:
                continue


            # Make pdf
            # results.canvas.Update()
            # results.canvas.SaveAs(pdf_path)

            hists = list()

            for i in results.root_artifacts:
                if isinstance(i, np.ndarray): #i.InheritsFrom('TH2') or i.InheritsFrom('TH1'):
                    hists.append(i)
                #if i.InheritsFrom('TH1'):
                #    hist.append(i)


            # Make png
            #subprocess.Popen(
            #    ['convert', '-density', '50', '-trim', '-fuzz', '1%', pdf_path, png_path])

            # Make json
            info = {
                'id': result_id,
                'name': hp.data_name,
                'comparator': comp_name,
                'display': results.show or hp.config.get('always_show', False),
                #'config': hp.config,
                #'results': results.info,
                'info': results.info,
                #'pdf_path': pdf_path,
                #'json_path': json_path,
                #'png_path': png_path,
                'artifacts': results.root_artifacts,
                'hists' : hists
            }
            # with open(json_path, 'w') as jf:
            #     json.dump(info, jf)


            hist_outputs.append(info)

    return hist_outputs

## get name of all th1 and th2 in a given directory
# def getall(d, h):
    
    ## f.allkeys(filterclass=lambda cls: issubclass(cls, uproot.rootio.uproot_methods.classes.TH1.Methods)) might work
    
    
#     keyName = d.GetDirectory(h)
#     for key in keyName.GetListOfKeys():
#         obj = keyName.Get(key.GetName())
#         ## this is so we don't return a null pointer
#         if obj:
#             objName = obj.GetName()
#             if obj.IsFolder():
#                 for i in getall(d, h+'/'+objName):
#                     yield i
#             else:
#                 if obj.InheritsFrom('TH1') or obj.InheritsFrom('TH2'):
#                     yield h+'/'+objName
#         else: 
            
#             continue

def compile_histpairs(config_dir, subsystem,
                      data_series, data_sample, data_run, data_path,
                      ref_series, ref_sample, ref_run, ref_path,
                      ref_list, ref_runs_list):

    config = cfg.get_subsystem(config_dir, subsystem)
    # Histogram details
    conf_list = config["hists"]
    main_gdir = config["main_gdir"]

    # ROOT files
    # data_file = ROOT.TFile.Open(data_path)
    # ref_file = ROOT.TFile.Open(ref_path)
    # ref_files_list = [ROOT.TFile.Open(x) for x in ref_list]
    data_file = uproot.open(data_path)
    ref_file = uproot.open(ref_path)
    ref_files_list = [uproot.open(x) for x in ref_list]


    histPairs = []
    
    # dqmhists = []
    # for k in getall(data_file, h=f'DQMData/Run {data_run}/L1T/Run summary'):
    #     dqmhists.append(k)
    histlist = []

    for hconf in conf_list:
        # Get name of hist in root file
        h = str(hconf["path"].split("/")[-1])
        # Get parent directory of hist
        
        gdir = str(hconf["path"].split(h)[0])

        data_dirname = "{0}{1}".format(main_gdir.format(data_run), gdir)
        ref_dirname = "{0}{1}".format(main_gdir.format(ref_run), gdir)
        ref_dirnames_list = ["{0}{1}".format(main_gdir.format(x), gdir) for x in ref_runs_list]
        
        
        # data_dir = data_file.GetDirectory(data_dirname)
        # ref_dir = ref_file.GetDirectory(ref_dirname)
        # ref_dirs_list = [x.GetDirectory(y) for x,y in zip(ref_files_list, ref_dirnames_list)]



        data_dir = data_file[data_dirname[:-1]]
        ref_dir = ref_file[ref_dirname[:-1]]
        ref_dirs_list = [x[y[:-1]] for x,y in zip(ref_files_list, ref_dirnames_list)]


        if not data_dir:
            raise error(
                "Subsystem dir {0} not found in data root file".format(data_dirname))
        if not ref_dir:
            raise error(
                "Subsystem dir {0} not found in ref root file".format(ref_dirname))
        for i in ref_dirs_list:
            if not i:
                raise error(
                "Subsystem dir {0} not found in ref root file".format(i))

        # data_keys = data_dir.GetListOfKeys()
        # ref_keys = ref_dir.GetListOfKeys()
        data_keys = [x.decode() for x in data_dir.keys()]
        ref_keys = [x.decode() for x in ref_dir.keys()]
        
        

        valid_names = []
    
    
        # # Add existing histograms that match h to valid_names
        # if "*" not in h:
        #     if data_keys.Contains(h) and ref_keys.Contains(h):
        #         valid_names.append(h)
        # else:
        #     # Check entire directory for files matching wildcard
        #     for name in [key.GetName() for key in data_keys]:
        #         if h.split("*")[0] in name and ref_keys.Contains(name):
        #             valid_names.append(name)
                        
        # # Load the histograms and create HistPairs
        # for name in valid_names:
        #     data_hist = data_dir.Get(name)
        #     ref_hist = ref_dir.Get(name)
        #     ref_hists_list = [x.Get(name) for x in ref_dirs_list]

        #     # This try/catch is a dirty way of checking that this objects are something plottable
        #     try:
        #         data_hist.SetDirectory(0)
        #         ref_hist.SetDirectory(0)
        #         for i in ref_hists_list:
        #             i.SetDirectory(0)
        #         histlist.append(f'{data_dirname}{name}')
        #     except:
        #         continue

        #     hPair = HistPair(hconf,
        #                      data_series, data_sample, data_run, name, data_hist,
        #                      ref_series, ref_sample, ref_run, name, ref_hist,
        #                      ref_runs_list, ref_hists_list)
        #     histPairs.append(hPair)
        
        # Add existing histograms that match h
        if "*" not in h:
             if h in [str(keys)[0:-2] for keys in data_keys] and h in [str(keys)[0:-2] for keys in ref_keys]:
                 try:
                     data_hist = data_dir[h]
                     ref_hist = ref_dir[h]
                 except Exception as e:
                     continue
                 ref_hists_list = [x[h] for x in ref_dirs_list]
                 hPair = HistPair(hconf,
                                  data_series, data_sample, data_run, str(h), data_hist,
                                  ref_series, ref_sample, ref_run, str(h), ref_hist, 
                                  ref_runs_list, ref_hists_list)
                 histPairs.append(hPair)
        else:
            # Check entire directory for files matching wildcard (Throw out wildcards with / in them as they are not plottable)
            for name in data_keys:

                if h.split("*")[0] in str(name) and name in ref_keys and not "<" in str(name):   
                    if ('/' not in name[:-2]):
                        try:
                            data_hist = data_dir[name[:-2]]
                            ref_hist = ref_dir[name[:-2]]
                        except Exception as e:
                            continue
                        ref_hists_list = [x[name[:-2]] for x in ref_dirs_list]
                        hPair = HistPair(hconf,
                                         data_series, data_sample, data_run, str(name[:-2]), data_hist,
                                         ref_series, ref_sample, ref_run, str(name[:-2]), ref_hist, 
                                         ref_runs_list, ref_hists_list)
                        histPairs.append(hPair)


    return histPairs


def load_comparators(plugin_dir):
    """Load comparators from each python module in ADQM_PLUGINS."""
    comparators = dict()


    for modname in os.listdir(plugin_dir):
        print(modname)

    for modname in os.listdir(plugin_dir):
        if modname[0] == '_' or modname[-4:] == '.pyc' or modname[0] == '.':
            continue
        if modname[-3:] == '.py':
            modname = modname[:-3]
        try:
            sys.path.append('/home/chosila/Projects/metaAnalysis/plugins')
            # mod = __import__(f"{modname}")
            if modname == 'ks':
                import ks as mod
            elif modname == 'pullvals':
                import pullvals as mod
            else:
                continue


            new_comps = mod.comparators()

        except AttributeError:
            raise error(
                "Plugin {} does not have a comparators() function.".format(mod))
        comparators.update(new_comps)

    return comparators


def identifier(hp, comparator_name):
    """Return a `hashed` identifier for the histpair"""
    data_id = "DATA-{}-{}-{}".format(hp.data_series,
                                     hp.data_sample, hp.data_run)
    ref_id = "REF-{}-{}-{}".format(hp.ref_series, hp.ref_sample, hp.ref_run)
    if hp.data_name == hp.ref_name:
        name_id = hp.data_name
    else:
        name_id = "DATANAME-{}_REFNAME-{}".format(hp.data_name, hp.ref_name)
    comp_id = "COMP-{}".format(comparator_name)

    hash_snippet = str(hash(hp))[-5:]

    idname = "{}_{}_{}_{}_{}".format(
        data_id, ref_id, name_id, comp_id, hash_snippet)
    return idname


class error(Exception):
    pass

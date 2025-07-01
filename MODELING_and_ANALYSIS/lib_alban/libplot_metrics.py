import pdb
import numpy as np
import glob
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from labellines import labelLine, labelLines
import seaborn as sns
import math
from statannot import add_stat_annotation
import pickle


def plot_overall_thouless(DICT_MATCHES, savename):
    dat = pd.DataFrame(DICT_MATCHES)

    fig = plt.figure(figsize=(6., 6))
    ax = sns.boxplot(data=dat, x='Ref', y='Thouless', linewidth=3, palette=colors_ref, width=0.85, fliersize=10)
    test_results = add_stat_annotation(ax, data=dat, x='Ref', y='Thouless',
                                       box_pairs=[(list_labels_ref[0], list_labels_ref[1]),
                                                  (list_labels_ref[1], list_labels_ref[2]),
                                                  (list_labels_ref[0], list_labels_ref[2])],
                                       test='t-test_paired', text_format='star',
                                       loc='outside', verbose=2)
    ax.set_ylabel('Thouless Ratio')
    ax.set_ylim(-1, 1.5)
    fig.subplots_adjust(top=0.973, bottom=0.049, left=0.139, right=0.955, hspace=0.2, wspace=0.2)
    fig.savefig('figures/boxplotThoulessVsRef%s.png'%savename)
    plt.close()


def plot_n_analyse_stats(illumref, illumtest, MATCHES,  list_labels_ref, rref, colors_ref, xpname=''):
    ### PLOTS and stats

    STATS_fit = {}
    STATS_fit['thouless'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['origin'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['linearity'] = np.zeros(MATCHES.shape[:2])

    # Range log reflectances to compute goodness of linear fit
    lin_range = np.abs(1 - 0)
    log_range = np.log(1) - np.log(0.01)

    size_subplots = np.sqrt(len(MATCHES))
    cols = math.ceil(size_subplots)
    rows = math.ceil(len(MATCHES) // size_subplots)
    if cols * rows < len(MATCHES):
        rows = rows + 1

    # import pdb; pdb.set_trace()
    fig, subs = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=(12, 12 * rows / cols))
    X = np.log(illumtest[0, :, :, 0].mean(0)) - np.log(illumref[0, :, :, 0].mean(0))
    x0 = X[0] - X[0] / 10
    x1 = X[-1] + X[-1] / 10
    Y = np.log(np.nanmean(MATCHES, axis = -1))

    reg = LinearRegression(fit_intercept=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # defualt colors used by matplotlib
    for p in range(len(MATCHES)):
        for i, r in enumerate(list_labels_ref):  # loop over references
            y = Y[p, i]
            for trial in range(len(MATCHES[p, i, 0])):
                subs[p // cols, p % cols].scatter(X, np.log(MATCHES[p, i, :, trial]), s=0.5, color=colors[i])
            subs[p // cols, p % cols].scatter(X, np.log(np.nanmean(MATCHES[p, i], axis = -1)), s=10, color=colors[i])
            reg.fit(X[:, np.newaxis], y)
            subs[p // cols, p % cols].plot([x0, x1], [reg.intercept_ + reg.coef_ * x0, reg.intercept_
                                                      + reg.coef_ * x1],
                                           label=str(1.00 + np.round(reg.coef_, 2)))
            subs[p // cols, p % cols].scatter(x0, np.log(rref[0, i].mean()), s=10, marker='s',
                                              color='k')  # plot dashed line for GT
            subs[p // cols, p % cols].set_xticks(X)
            if 'Human' in xpname:
                subs[p // cols, p % cols].set_title('Observer %i' % p)
            else:
                subs[p // cols, p % cols].set_title('Model Instance %i' % p)
            subs[p // cols, p % cols].set_xlabel('log Illuminations')

            ### save stats fit
            # import pdb; pdb.set_trace()

            STATS_fit['thouless'][p, i] = 1.00 + np.round(reg.coef_, 2)

            mean_residuals = np.mean((reg.predict(X[:, np.newaxis]) - y) ** 2)
            STATS_fit['linearity'][p, i] = 1 - np.sqrt(mean_residuals) / (0.25 * log_range)

            STATS_fit['origin'][p, i] = 1 - np.abs(reg.intercept_ - np.log(rref[0, i].mean())) / (0.25 * log_range)

        labelLines(subs[p // cols, p % cols].get_lines())
    # Formating
    for c in range(rows):
        subs[c, 0].set_ylabel('log Match')
    plt.tight_layout()
    plt.show()
    # save path
    instance = xpname.split('_')[0]
    condition = xpname.split('_')[-1][:-4]
    #import pdb; pdb.set_trace()
    fig.savefig(f'figures/{instance}/thouless_{condition}].png')
    plt.close()

    ### Figures stats thouless

    fig, subs = plt.subplots(1, 3, figsize=(10, 4))
    dat = STATS_fit['thouless']
    subs[0].set_title('Thouless')
    sns.boxplot(ax=subs[0], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = STATS_fit['origin']
    subs[1].set_title('$\Delta$0')
    sns.boxplot(ax=subs[1], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = STATS_fit['linearity']
    subs[2].set_title('Linearity')
    sns.boxplot(ax=subs[2], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    subs[0].set_ylabel('Thouless ratio', fontsize=14)
    subs[0].set_ylim(0, 1.25)
    subs[1].set_ylim(0, 1.25)
    subs[2].set_ylim(0, 1.25)
    for i in range(3):
        labelsTicks = [item.get_text() for item in subs[i].get_xticklabels()]
        for j in range(len(list_labels_ref)):
            labelsTicks[j] = '%s' % str(list_labels_ref[j])
        #subs[i].set_xticklabels(labelsTicks, fontsize=14)
    plt.tight_layout()
    plt.show()
    #fig.savefig(f'figures/{instance}/BoxplotThoulessVsLabel_{condition}.png')
    plt.close()
    return STATS_fit

def plot_n_analyse_stats_one(illumref, illumtest, MATCHES,  list_labels_ref, rref, colors_ref, xpname=''):
    ### PLOTS and stats

    STATS_fit = {}
    STATS_fit['thouless'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['origin'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['linearity'] = np.zeros(MATCHES.shape[:2])

    # Range log reflectances to compute goodness of linear fit
    lin_range = np.abs(1 - 0)
    log_range = np.log(1) - np.log(0.01)

    size_subplots = np.sqrt(len(MATCHES))

    # import pdb; pdb.set_trace()
    fig, subs = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(6, 6))
    X = np.log(illumtest[0, :, :, 0].mean(0)) - np.log(illumref[0, :, :, 0].mean(0))
    x0 = X[0] - X[0] / 10
    x1 = X[-1] + X[-1] / 10
    Y = np.log(MATCHES.mean(-1))

    reg = LinearRegression(fit_intercept=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # defualt colors used by matplotlib
    for i, r in enumerate(list_labels_ref):  # loop over references
        y = Y[0, i]
        for trial in range(len(MATCHES[0, i, 0])):
            subs.scatter(X, np.log(MATCHES[0, i, :, trial]), s=0.5, color=colors[i])
        subs.scatter(X, np.log(MATCHES[0, i].mean(-1)), s=10, color=colors[i])
        reg.fit(X[:, np.newaxis], y)
        subs.plot([x0, x1], [reg.intercept_ + reg.coef_ * x0, reg.intercept_
                                                  + reg.coef_ * x1],
                                       label=str(1.00 + np.round(reg.coef_, 2)))
        subs.scatter(x0, np.log(rref[0, i].mean()), s=10, marker='s',
                                          color='k')  # plot dashed line for GT
        subs.set_xticks(X)
        subs.set_xticklabels(np.exp(X))
        if 'Human' in xpname:
            subs.set_title('Observer ')
        else:
            subs.set_title('Model Instance')
        subs.set_xlabel('log Illuminations')


        STATS_fit['thouless'][0, i] = 1.00 + np.round(reg.coef_, 2)

        mean_residuals = np.mean((reg.predict(X[:, np.newaxis]) - y) ** 2)
        STATS_fit['linearity'][0, i] = 1 - np.sqrt(mean_residuals) / (0.25 * log_range)

        STATS_fit['origin'][0, i] = 1 - np.abs(reg.intercept_ - np.log(rref[0, i].mean())) / (0.25 * log_range)

    labelLines(subs.get_lines())
    # Formating
    subs.set_ylabel('log Match')
    plt.tight_layout()
    plt.show()
    # save path
    instance = xpname.split('_')[0]
    condition = xpname.split('_')[-1][:-4]
    fig.savefig(f'figures/{instance}/thouless_{condition}].png')
    plt.close()

    ### Figures stats thouless

    fig, subs = plt.subplots(1, 3, figsize=(10, 4))
    dat = STATS_fit['thouless']
    subs[0].set_title('Thouless')
    sns.boxplot(ax=subs[0], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = STATS_fit['origin']
    subs[1].set_title('$\Delta$0')
    sns.boxplot(ax=subs[1], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = STATS_fit['linearity']
    subs[2].set_title('Linearity')
    sns.boxplot(ax=subs[2], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    subs[0].set_ylabel('Thouless ratio', fontsize=14)
    subs[0].set_ylim(0, 1.25)
    subs[1].set_ylim(0, 1.25)
    subs[2].set_ylim(0, 1.25)
    for i in range(3):
        labelsTicks = [item.get_text() for item in subs[i].get_xticklabels()]
        for j in range(len(list_labels_ref)):
            labelsTicks[j] = '%s' % str(list_labels_ref[j])
        subs[i].set_xticklabels(labelsTicks, fontsize=14)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'figures/{instance}/BoxplotThoulessVsLabel_{condition}.png')
    plt.close()
    return STATS_fit

def plot_linReg(illumref, illumtest, MATCHES, list_labels_ref, rref, xpname='', p=0, lr=0):# p for participant, default 0 if there's only one model;

    model = xpname.split('_')[0]
    condition = xpname.split('_')[1]
    fig, sub = plt.subplots(1, 1, figsize=(5.5, 5))
    X = np.absolute(np.log(illumtest[0, :, :, lr].mean(0)) - np.log(illumref[0, :, :, lr].mean(0))) # x axis
    x0 = X[0] - X[0] / 10
    x1 = X[-1] + X[-1] / 10
    Y = np.log(np.nanmean(MATCHES, axis = -1))

    reg = LinearRegression(fit_intercept=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # defualt colors used by matplotlib
    for i, r in enumerate(list_labels_ref):  # loop over references
        y = Y[p, i]
        for trial in range(len(MATCHES[p, i, 0])):
            sub.scatter(X, np.log(MATCHES[p, i, :, trial]), s=4, color=colors[i])
        sub.scatter(X, np.log(np.nanmean(MATCHES[p, i], axis = -1)), s=30, color=colors[i])
        reg.fit(X[:, np.newaxis], y)
        sub.plot([x0, x1], [reg.intercept_ + reg.coef_ * x0, reg.intercept_
                                                  + reg.coef_ * x1],
                                       label=str(1.00 + np.round(reg.coef_, 2)), lw=2)
        sub.scatter(x0, np.log(rref[0, i].mean()), s=25, marker='s',
                                          color='k')  # plot dashed line for GT
        xlab = np.array([1.0, 1.5, 2.0, 3.0, 5.0])
        sub.set_xticks(np.log(xlab))
        sub.set_xticklabels(xlab)
        ylab = np.array([0.05, 0.1, 0.2, 0.4, 0.6])
        sub.set_yticks(np.log(ylab))
        sub.set_yticklabels(ylab)
        if 'Human' in xpname:
            sub.set_title('Observer %i' % p)
        else:
            sub.set_title('Model Instance %i' % p)


    labelLines(sub.get_lines())
    # Formating
    plt.ylim(-3.5, -0.25)
    # plt.title('Observer %i' % p)
    plt.xlabel('Match illuminance (log scale)', fontsize = 17)
    #plt.ylim(ylim)

    plt.ylabel('Match albedo (log scale)', fontsize = 17)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'figures/{model}/{condition}_linreg.png')
    plt.show()
    plt.close()


def plot_conditions(illumref, illumtest, MATCHES, list_labels_ref, rref, colors_ref, conditions, xpname='', cube = False):
    ### PLOTS and stats
    instance = xpname.split('_')[1]
    STATS_fit = {}
    STATS_fit['thouless'] = {}
    STATS_fit['origin'] = {}
    STATS_fit['linearity'] = {}
    #conditions = ['normal', 'cube', 'floor', 'sphere', 'floorsphere', 'whole']
    for condition in conditions:
        STATS_fit['thouless'][condition] = np.zeros(MATCHES[condition].shape[:2])
        STATS_fit['origin'][condition] = np.zeros(MATCHES[condition].shape[:2])
        STATS_fit['linearity'][condition] = np.zeros(MATCHES[condition].shape[:2])

        # Range log reflectances to compute goodness of linear fit
        lin_range = np.abs(1 - 0)
        log_range = np.log(1) - np.log(0.01)

        X = np.log(illumtest[condition][0, :, :, 0].mean(0)) - np.log(illumref[condition][0, :, :, 0].mean(0))
        x0 = X[0] - X[0] / 10
        x1 = X[-1] + X[-1] / 10
        Y = np.log(np.nanmean(MATCHES[condition], axis = -1))

        reg = LinearRegression(fit_intercept=True)
        for p in range(len(MATCHES[condition])):
            for i, r in enumerate(list_labels_ref):  # loop over references
                y = Y[p, i]
                reg.fit(X[:, np.newaxis], y)
                ### save stats fit
                # import pdb; pdb.set_trace()

                STATS_fit['thouless'][condition][p, i] = 1.00 + np.round(reg.coef_, 2)

                mean_residuals = np.mean((reg.predict(X[:, np.newaxis]) - y) ** 2)
                STATS_fit['linearity'][condition][p, i] = 1 - np.sqrt(mean_residuals) / (0.25 * log_range)

                STATS_fit['origin'][condition][p, i] = 1 - np.abs(reg.intercept_ - np.log(rref[0, i].mean())) / (
                            0.25 * log_range)

        STATS_fit['thouless'][condition] = STATS_fit['thouless'][condition].mean(-1)
        STATS_fit['linearity'][condition] = STATS_fit['linearity'][condition].mean(-1)
        STATS_fit['origin'][condition] = STATS_fit['origin'][condition].mean(-1)
    ### Figures stats thouless

    fig, subs = plt.subplots(1, 3, figsize=(15, 4))
    dat = pd.DataFrame.from_dict(STATS_fit['thouless'])
    subs[0].set_title('Thouless')
    sns.boxplot(ax=subs[0], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = pd.DataFrame.from_dict(STATS_fit['origin'])
    subs[1].set_title('$\Delta$0')
    sns.boxplot(ax=subs[1], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    dat = pd.DataFrame.from_dict(STATS_fit['linearity'])
    subs[2].set_title('Linearity')
    sns.boxplot(ax=subs[2], data=dat, palette=colors_ref, linewidth=3, width=0.85, fliersize=10)
    subs[0].set_ylabel('Thouless ratio', fontsize=15)
    subs[0].set_ylim(0, 1.2)
    subs[1].set_ylim(0, 1.2)
    subs[2].set_ylim(0, 1.2)
    #conditions_tick = ['Normal', 'Cube', 'Floor', 'Sphere', 'Floor &\nSphere', 'No cue']
    if conditions == ['normal', 'cube', 'sphere', 'floor', 'floorsphere', 'whole']:
        conditions_tick = ['Normal', 'Contrast', 'Shading', 'Shadows', 'Shading &\nshadows', 'No cue']
    elif conditions == ['normal', 'sphere', 'floor', 'floorsphere', 'whole']:
        conditions_tick = ['Normal', 'Contrast', 'Shading', 'Shadows', 'Shading &\nshadows', 'No cue']
    else:
        conditions_tick = ['Normal', 'No cue']
    for i in range(len(subs)):
        labelsTicks = [item.get_text() for item in subs[i].get_xticklabels()]
        for j in range(len(conditions_tick)):
            labelsTicks[j] = '%s' % str(conditions_tick[j])
        subs[i].set_xticks(range(len(labelsTicks)))
        subs[i].set_xticklabels(labelsTicks, fontsize=14)
    plt.tight_layout()
    plt.show()
    fig.savefig(f'figures/{instance}/Boxplot_conditions_{xpname}.png', dpi=600)
    plt.close()

    nb_models = len(STATS_fit['thouless']['normal'])
    ARRAY_thouless = np.zeros((nb_models, len(conditions)))
    fig, subs = plt.subplots(1, 1, figsize=(3, 8)) # 15,8 for eevee, 11,8 for cycle models, 3,8 for human cycls
    for c, condition in enumerate(conditions):
        ARRAY_thouless[:,c] = STATS_fit['thouless'][condition]
    subs.plot(np.arange(1,len(conditions) + 1), ARRAY_thouless.T, color = 'grey', lw=3)
    #subs.plot(np.arange(1, len(conditions) + 1), ARRAY_thouless[8].T, color='grey', lw=3)
    #subs.plot(np.arange(1, len(conditions) + 1), ARRAY_thouless[7].T, color='grey', lw=3, ls = ':')
    #subs.plot(np.arange(1, len(conditions) + 1), ARRAY_thouless[0].T, color='grey', lw=3, ls=':')
    mean_thouless = (ARRAY_thouless.T).mean(-1)
    print(f'Mean thouless ratios are: {mean_thouless}')
    subs.plot(np.arange(1, len(conditions) + 1), mean_thouless, color='k', lw=10)
    # conditions_tick = ['Normal', 'Cube', 'Floor', 'Sphere', 'Floor &\nSphere', 'No cue']
    plt.xticks(np.arange(1,len(conditions) + 1))
    plt.tick_params(axis='both', which='major', labelsize=13)
    labelsTicks = [item.get_text() for item in subs.get_xticklabels()]
    for j in range(len(conditions_tick)):
        labelsTicks[j] = '%s' % str(conditions_tick[j])
    subs.set_xticklabels(labelsTicks, fontsize=17)
    plt.ylabel('Thouless ratio', fontsize=17)
    plt.ylim((-0.05, 1.05))
    plt.xlim(0.8, len(conditions) + 0.2)
    plt.tight_layout()
    plt.show()
    print(f'figures/{instance}/Curves_conditions_{xpname}.png')
    fig.savefig(f'figures/{instance}/Curves_conditions_{xpname}.png')
    plt.close()

    return fig, subs, mean_thouless



def plot_thouless(illumref, illumtest, MATCHES, list_labels_ref, rref, xpname='', p=0, lr=0, r = 0):# p for participant, default 0 if there's only one model;
    fig, sub = plt.subplots(1, 1, figsize=(7, 6))
    X = np.log(illumtest[p, :, :, lr].mean(0)) - np.log(illumref[p, :, :, lr].mean(0)) # x axis
    x0 = X[0] - X[0] / 10
    x1 = X[-1] + X[-1] / 10
    Y = np.log(MATCHES[:,:,:,lr])

    reg = LinearRegression(fit_intercept=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # defualt colors used by matplotlib
    y = Y[p, r]
    for trial in range(len(MATCHES[p, r, 0])):
        sub.scatter(X, np.log(MATCHES[p, r, :, trial]), s=0.5, color=colors[r])
    sub.scatter(X, np.log(MATCHES[p, r].mean(-1)), s=10, color=colors[r])
    reg.fit(X[:, np.newaxis], y)
    sub.plot([x0, x1], [reg.intercept_ + reg.coef_ * x0, reg.intercept_
                                              + reg.coef_ * x1],
                                   label=str(1.00 + np.round(reg.coef_, 2)))
    sub.scatter(x0, np.log(rref[p, r].mean()), s=10, marker='s',
                                      color='k')  # plot dashed line for GT
    sub.set_xticks(X)
    sub.set_xlabel('log Illuminations')

    labelLines(sub.get_lines())
    # Formating

    sub.set_ylabel('log Match')
    plt.tight_layout()

    plt.xticks(X)
    # plt.title('Observer %i' % p)
    plt.xlabel('log Illuminations')
    #plt.ylim(ylim)
    plt.ylabel('log matched reflectance')
    plt.show()
    fig.savefig('figures/%s_linreg.png' % xpname)
    plt.show()
    plt.close()

def plot_n_analyse_stats_channels(illumref, illumtest, MATCHES, list_labels_ref, rref, colors_ref, xpname=''):
    ### PLOTS and stats

    STATS_fit = {}
    STATS_fit['thouless'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['origin'] = np.zeros(MATCHES.shape[:2])
    STATS_fit['linearity'] = np.zeros(MATCHES.shape[:2])

    # Range log reflectances to compute goodness of linear fit
    lin_range = np.abs(1 - 0)
    log_range = np.log(1) - np.log(0.01)

    size_subplots = np.sqrt(len(MATCHES))
    cols = math.ceil(size_subplots)
    rows = math.ceil(len(MATCHES) // size_subplots)
    if cols * rows < len(MATCHES):
        rows = rows + 1

    # import pdb; pdb.set_trace()
    fig, subs = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=(12, 12 * rows / cols))
    X = np.log(illumtest[0, :, :, 0].mean(0)) - np.log(illumref[0, :, :, 0].mean(0))
    x0 = X[0] - X[0] / 10
    x1 = X[-1] + X[-1] / 10
    Y = np.log(MATCHES.mean(-1))

    reg = LinearRegression(fit_intercept=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']  # defualt colors used by matplotlib
    for p in range(len(MATCHES)):
        for i, r in enumerate(list_labels_ref):  # loop over references
            y = Y[p, i]
            for trial in range(len(MATCHES[p, i, 0])):
                subs[p // cols, p % cols].scatter(X, np.log(MATCHES[p, i, :, trial]), s=0.5, color=colors[i])
            subs[p // cols, p % cols].scatter(X, np.log(MATCHES[p, i].mean(-1)), s=10, color=colors[i])
            reg.fit(X[:, np.newaxis], y)
            subs[p // cols, p % cols].plot([x0, x1], [reg.intercept_ + reg.coef_ * x0, reg.intercept_
                                                      + reg.coef_ * x1],
                                           label=str(1.00 + np.round(reg.coef_, 2)))
            subs[p // cols, p % cols].scatter(x0, np.log(rref[p, i].mean()), s=10, marker='s',
                                              color='k')  # plot dashed line for GT
            subs[p // cols, p % cols].set_xticks(X)
            if 'Human' in xpname:
                subs[p // cols, p % cols].set_title('Observer %i' % p)
            else:
                subs[p // cols, p % cols].set_title('Model Instance %i' % p)
            subs[p // cols, p % cols].set_xlabel('log Illuminations')

            ### save stats fit
            # import pdb; pdb.set_trace()

            STATS_fit['thouless'][p, i] = 1.00 + np.round(reg.coef_, 2)

            mean_residuals = np.mean((reg.predict(X[:, np.newaxis]) - y) ** 2)
            STATS_fit['linearity'][p, i] = 1 - np.sqrt(mean_residuals) / (0.25 * log_range)

            STATS_fit['origin'][p, i] = 1 - np.abs(reg.intercept_ - np.log(rref[p, i].mean())) / (0.25 * log_range)

        labelLines(subs[p // cols, p % cols].get_lines())
    # Formating
    for c in range(rows):
        subs[c, 0].set_ylabel('log Match')
    plt.tight_layout()
    plt.show()
    # save path
    instance = xpname.split('_')[1]
    condition = xpname.split('_')[-1][:-4]
    fig.savefig(f'figures/{instance}/thouless_{condition}].png')
    plt.close()

    return STATS_fit
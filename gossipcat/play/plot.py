#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## ========================
## time series
## ========================
def plot_ts(ts, value, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(30, 8))

    ax.plot(ts, value, color='gray')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return None

def extract_ts(df, ts_col, startdate, enddate):
    startdate = pd.to_datetime(startdate).date()
    enddate = pd.to_datetime(enddate).date()
    df = df[(df[ts_col] > startdate) & (df[ts_col] <= enddate)]
    return df

def plot_ts_period(df, ts_col, value_col, xlabel, ylabel, title, startdate, enddate):
    df = extract_ts(df, ts_col, startdate, enddate)
    plot_ts(ts = df[ts_col], 
            value = df[value_col], 
            xlabel = xlabel,
            ylabel = ylabel,
            title = title+' (%s to %s)' %(startdate, enddate))
    plt.show()
    return None

## ========================
## bar plot
## ========================
def barplot(x, labels, xlabel, title):
    fig, ax = plt.subplots(figsize=(6, 10))

    ax.barh(labels, x, align='center')
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    plt.show()
    return None

## ========================
## donut plot
## ========================
def plot_donut(x, labels, title):
    plt.figure(figsize=(6, 6))
    plt.pie(x = x, 
            labels = labels, 
            wedgeprops = {'linewidth': 7, 'edgecolor': 'white'})

    my_circle=plt.Circle((0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.title(title)
    plt.show()
    return None

## ========================
## calendar heatmap 
## ========================
def heatmap_calendar(df, ts_col, value_col, title, startdate, enddate, cmap=plt.cm.RdBu_r):
    import july
    df = extract_ts(df, ts_col, startdate, enddate)

    july.heatmap(df[ts_col], 
                 df[value_col], 
                 title=title, 
                 cmap=cmap)
    return None

## ========================
## timeline
## ========================
def plot_timeline(dates, names, title, levels=10):
    # Choose some nice levels
    l = []
    for i in range(1, levels): 
        l.append(i)
        l.append(-i)
    levels = np.tile(l, int(np.ceil(len(dates)/6)))[:len(dates)]

    # Create figure and plot a stem plot with the date
    fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
    ax.set(title=title)

    ax.vlines(dates, 0, levels, color="tab:red")  # The vertical stems.
    ax.plot(dates, np.zeros(dates.size), "-o",
            color="k", markerfacecolor="w")  # Baseline and markers on it.

    # annotate lines
    for d, l, r in zip(dates, levels, names):
        ax.annotate(r, xy=(d, l),
                    xytext=(-3, np.sign(l)*3), textcoords="offset points",
                    horizontalalignment="right",
                    verticalalignment="bottom" if l > 0 else "top")

    # format xaxis with 4 month intervals
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # remove y axis and spines
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.margins(y=0.1)
    plt.show()
    return None

## ========================
## word cloud
## ========================
def extract_word_count(df, col_txt):
    import jieba.posseg as pseg
    content = df[col_txt].values.tolist()
    segment=[]
    for line in content:
        try:
            segs = pseg.cut(line, use_paddle=True)
            for seg in segs:
                segment.append((seg.word, seg.flag))
        except:
            print(line)
            continue
    counter = collections.Counter(segment)
    df_words = pd.DataFrame(counter.keys(), columns=['word', 'type'])
    df_words['cnt'] = counter.values()
    df_words = df_words.sort_values('cnt', ascending=False)
    return df_words

def word_cloud(df, col_word, col_cnt, max_words, mask_pic, title):
    import numpy as np
    import matplotlib as mpl
    from PIL import Image
    from wordcloud import WordCloud
    mask = np.array(Image.open(mask_pic))
    font_path="/System/Library/fonts/PingFang.ttc"
    
    cmap = mpl.cm.Blues(np.linspace(0,1,20)) 
    cmap = mpl.colors.ListedColormap(cmap[-10:,:-1]) 

    df = df[[col_word, col_cnt]]
    frequency = {x[0]:x[1] for x in df.values}
    plt.figure(figsize=(20, 10), linewidth=12, edgecolor="red")
    
    wc = WordCloud(font_path=font_path,
                   scale=3,
                   mask=mask,
                   colormap=cmap,
                   background_color='white',
                   contour_color='red',
                   contour_width=8).generate_from_frequencies(frequency)
    
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.title(title, fontsize=35, fontweight='bold', color='white', backgroundcolor= 'red', pad=10)
    plt.show()
    return None

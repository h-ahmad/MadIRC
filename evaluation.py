# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:44:52 2024

@author: Hussain Ahmad Madni
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


def association_depth_auroc(color1, color2, size):
    assoc_depth = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    img_auroc = [0.8410, 0.8430, 0.8460, 0.8490, 0.8515, 0.8502, 0.8520, 0.8550, 0.8550]
    pix_auroc = [0.8890, 0.8958, 0.9017, 0.9079, 0.9166, 0.9205, 0.9226, 0.92385, 0.9240]
    plt.rcParams['figure.figsize'] = size
    plt.plot(assoc_depth, img_auroc, label = 'Image AUROC', color=color1, lw=1, linestyle='--', marker='o')
    plt.plot(assoc_depth, pix_auroc, label = 'Pixel AUROC', color=color2, lw=1, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('Association Depth', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    #plt.title('Association Depth vs Image and Pixel AUROC', fontsize=12)
    plt.tight_layout()
    plt.savefig('association_depth_auroc.pdf')
    plt.show()
    
def association_depth_precision_recall(color1, color2, size):
    assoc_depth = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    precision = [0.99900, 0.99895, 0.99882, 0.99874, 0.99867, 0.99856, 0.99847, 0.99834, 0.99826]
    recall = [0.28908, 0.28908, 0.28907, 0.28904, 0.28901, 0.28894, 0.28890, 0.28888, 0.28882]
    plt.rcParams['figure.figsize'] = size
    fig, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot(assoc_depth, precision, label = 'Tube Precision', color=color1, lw=1, linestyle='-', marker='o')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.set_xlabel('Association Depth', fontsize=12)
    ax1.set_ylabel('Tube Precision', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(assoc_depth, recall, label = 'Tube Recall', color=color2, lw=1, linestyle='-', marker='o')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.set_ylabel("Tube Recall", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.xticks([0, 10, 20, 30, 40])
    plt.tight_layout()
    plt.savefig('association_depth_precision_recall.pdf')
    plt.show()
    
def maximum_channel_auroc(color1, color2, size):
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    img_auroc = [0.8410, 0.8430, 0.8462, 0.8493, 0.8515, 0.8520, 0.8550, 0.8554, 0.8553, 0.8553, 0.8553, 0.8553, 0.8552, 0.8551, 0.8550]
    pix_auroc = [0.8890, 0.8958, 0.9017, 0.9079, 0.9166, 0.9205, 0.9226, 0.92385, 0.9240, 0.9204, 0.9180, 0.9120, 0.9040, 0.8980, 0.8940]
    plt.rcParams['figure.figsize'] = size
    plt.plot(assoc_depth, img_auroc, label = 'Image AUROC', color=color1, lw=1, linestyle='--', marker='o')
    plt.plot(assoc_depth, pix_auroc, label = 'Pixel AUROC', color=color2, lw=1, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('Maximum Channel Variance ($\sigma$)', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('maximum_channel_auroc.pdf')
    plt.show()
    
def maximum_channel_precision_recall(color1, color2, size):  
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    precision = [0.9990, 0.9989, 0.9987, 0.9986, 0.9984, 0.9982, 0.9981, 0.99807, 0.99806, 0.99805, 0.99804, 0.99803, 0.99802, 0.99801, 0.99800]
    recall = [0.28900, 0.28913, 0.28922, 0.28936, 0.28941, 0.28958, 0.28960, 0.28977, 0.28986, 0.28993, 0.29009, 0.29013, 0.29028, 0.29032, 0.29048]
    plt.rcParams['figure.figsize'] = size
    fig, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot(assoc_depth, precision, label = 'Tube Precision', color=color1, lw=1, linestyle='-', marker='o')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.set_xlabel('Maximum Channel Variance ($\sigma$)', fontsize=12)
    ax1.set_ylabel('Tube Precision', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(assoc_depth, recall, label = 'Tube Recall', color=color2, lw=1, linestyle='-', marker='o')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.set_ylabel("Tube Recall", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('maximum_channel_precision_recall.pdf')
    plt.show()
    
def minimum_channel_auroc(color1, color2, size):
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    img_auroc = [0.8554, 0.8539, 0.8525, 0.8515, 0.8490, 0.8471, 0.8443, 0.8415, 0.8397, 0.8378, 0.8356, 0.8334, 0.8322, 0.8311, 0.8280]
    pix_auroc = [0.9240, 0.9220, 0.9195, 0.9168, 0.9149, 0.9133, 0.9114, 0.9096, 0.9070, 0.9058, 0.9018, 0.9002, 0.8977, 0.8964, 0.8940]
    plt.rcParams['figure.figsize'] = size
    plt.plot(assoc_depth, img_auroc, label = 'Image AUROC', color=color1, lw=1, linestyle='--', marker='o')
    plt.plot(assoc_depth, pix_auroc, label = 'Pixel AUROC', color=color2, lw=1, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('Minimum Channel Span', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('minimum_channel_auroc.pdf')
    plt.show()

def minimum_channel_precision_recall(color1, color2, size):
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    precision = [0.99802, 0.99812, 0.99823, 0.99832, 0.99843, 0.99855, 0.99863, 0.99872, 0.99884, 0.99893, 0.99900, 0.99898, 0.99886, 0.99877, 0.99869]
    recall = [0.29042, 0.29031, 0.29029, 0.29018, 0.29004, 0.28997, 0.28985, 0.28972, 0.28964, 0.28952, 0.28947, 0.28935, 0.28928, 0.28917, 0.28909]
    plt.rcParams['figure.figsize'] = size
    fig, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot(assoc_depth, precision, label = 'Tube Precision', color=color1, lw=1, linestyle='-', marker='o')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.set_xlabel('Minimum Channel Span', fontsize=12)
    ax1.set_ylabel('Tube Precision', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(assoc_depth, recall, label = 'Tube Recall', color=color2, lw=1, linestyle='-', marker='o')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.set_ylabel("Tube Recall", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('minimum_channel_precision_recall.pdf')
    plt.show()
    
def positional_embedding_auroc(color1, color2, size):
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    img_auroc = [0.8500, 0.8505, 0.8510, 0.8513, 0.8520, 0.85227, 0.8536, 0.8547, 0.8561, 0.8555, 0.8532, 0.8521, 0.8522, 0.8524, 0.8526]
    pix_auroc = [0.8940, 0.8970, 0.8990, 0.9010, 0.9040, 0.9065, 0.9080, 0.9085, 0.9092, 0.9102, 0.9090, 0.9088, 0.9087, 0.9086, 0.9085]
    plt.rcParams['figure.figsize'] = size
    plt.plot(assoc_depth, img_auroc, label = 'Image AUROC', color=color1, lw=1, linestyle='--', marker='o')
    plt.plot(assoc_depth, pix_auroc, label = 'Pixel AUROC', color=color2, lw=1, linestyle='--', marker='o')
    plt.legend()
    plt.xlabel('Strength of Positional Embedding', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('positional_embedding_auroc.pdf')
    plt.show()
    
def positional_embedding_precision_recall(color1, color2, size):
    assoc_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    precision = [0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987, 0.9987]
    recall = [0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897, 0.2897]
    plt.rcParams['figure.figsize'] = size
    fig, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot(assoc_depth, precision, label = 'Tube Precision', color=color1, lw=1, linestyle='-', marker='o')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax1.set_xlabel('Strength of Positional Embedding', fontsize=12)
    ax1.set_ylabel('Tube Precision', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.plot(assoc_depth, recall, label = 'Tube Recall', color=color2, lw=1, linestyle='-', marker='o')
    ax2.set_ylim(0.2890, 0.2910)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax2.set_ylabel("Tube Recall", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.xticks([1, 5, 10, 15])
    plt.tight_layout()
    plt.savefig('positional_embedding_precision_recall.pdf')
    plt.show()

if __name__ == '__main__':
    color1 = 'red'
    color2 = 'blue'
    size1 = [4, 4]
    size2 = [6, 6]
    size3 = [8, 8]
    
    association_depth_auroc(color1, color2, size1)
    association_depth_precision_recall(color1, color2, size1)
    
    maximum_channel_auroc(color1, color2, size1)
    maximum_channel_precision_recall(color1, color2, size1)
    
    minimum_channel_auroc(color1, color2, size1)
    minimum_channel_precision_recall(color1, color2, size1)
    
    positional_embedding_auroc(color1, color2, size1)
    positional_embedding_precision_recall(color1, color2, size1)
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
from config import GetConfig
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from numpy import ma

def process (input_image, params, model_params, config):

    limbSeq = [list(x) for x in config.limbs_conn]
    numLimbConn = len(limbSeq)
    numParts = config.num_parts
    numPartsWithBackground = config.num_parts_with_background
    mapIdx = [[x * 2, x * 2 + 1] for x in range(numLimbConn)]

    
    cmap = get_cmap(numParts)
    colors = [cmap(i) for i in range(numParts)]
    colors = [[int(r * 255), int(g * 255), int(b * 255)] for (r, g, b, a) in colors]

    # read the input image
    oriImg = cv2.imread(input_image)  # B,G,R order

    # list of float multipliers used for scaling
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    # array to store the avg heatmap
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], numPartsWithBackground))

    # array to store avg part affinity field
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 2 * numLimbConn))

    # iterate over the different scales
    for m in range(len(multiplier)):


        scale = multiplier[m]

        # resize the image to the given scale
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # pad the lower right side of the image so it matches the stride
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])

        # get network's predictions based on the input image
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)

        # predict on the input image
        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding

        # heatmap comes out reduced from og size by 'stride' size and scale size
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    
        # resize the heatmap to og image size + padding
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)

        # remove the padding from the heatmap
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        
        # scale the heatmap to the original size
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # do the same scaling on the part affinity field
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)


        # sum and save in heatmap avg
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)


    # MY CODE VISUALIZE HEAT
    save_all_heat(oriImg, heatmap_avg, numParts, config)
    save_all_paf(oriImg, paf_avg, mapIdx, config)
    # for partIdx in range(numParts):
    #     visualize_heat(oriImg, heatmap_avg, partIdx, config)

    # MY CODE VISUALIZE PAF
    # for _, fromTO in enumerate(mapIdx):
    #     visualize_paf(oriImg, paf_avg, fromTO, config)

    # here heatmap_avg and paf_avg are same size as original image

    # each list is for a single part
    # all_peaks is a list of lists [(col, row, score, id), (col, row, score, id)] of peaks and scores and unique ids
    all_peaks = []
    # used to generate unique ids for peaks
    peak_counter = 0

    # iterate over the parts
    for part in range(numParts):

        # get the avg heatmap associated with the part
        map_ori = heatmap_avg[:, :, part]

        # smooth the avg heatmap associated with the part
        map = gaussian_filter(map_ori, sigma=3)

        # get the local peaks from the average heatmap (True, False array where True is peak, same size as og img)
        peaks_binary = peak_local_max(map, min_distance=1, threshold_abs=params['thre1'], indices=False)

        # peaks is a list [[col, row], [col, row]] of peak coordinates
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse

        # peaks with score is a list [(col, row, score), (col, row, score)] of peaks and scores
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

        # unique ids for peaks
        id = range(peak_counter, peak_counter + len(peaks))

        # peaks_with_score_and_id is a list [(col, row, score, id), (col, row, score, id)] of peaks and scores and unique ids
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        # store peaks and update peak counter
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # TODO define what these are

    # a list containing k arrays vertically stacked with 
    # [from peakID, toPeakId, score_with_dist_prior, all_peaks[limbSeq[k][0]] index, all_peaks[limbSeq[k][1]] index]
    # if there are no candidates for either from, or to, or both then we add the connection index k to special k
    # and we append an empty list to connection_all
    connection_all = []
    special_k = []
    mid_num = params['mid_num']

    # iterate over xy mapping in mapIdx 
    for k in range(len(mapIdx)):

        # score_mid [0] is the x vector
        # score_mid [1] is the y vector
        score_mid = paf_avg[:, :, mapIdx[k]]

        # candA is a list [(col, row, score, id), (col, row, score, id)] of peaks and scores and unique ids for the from part
        candA = all_peaks[limbSeq[k][0]]
        # candB is a list [(col, row, score, id), (col, row, score, id)] of peaks and scores and unique ids for the to part
        candB = all_peaks[limbSeq[k][1]]

        # get the number of candidates for a and b
        nA = len(candA)
        if nA == 0:
            print(f"no peaks for {config.parts[limbSeq[k][0]]}")

        nB = len(candB)
        if nB == 0:
            print(f"no peaks for {config.parts[limbSeq[k][1]]}")
        # indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            # i is index into candA
            for i in range(nA):
                # j is index into candB
                for j in range(nB):

                    # B - A => TO - From => vec from -> to
                    # vec is a vector [col, row] pointing in the direction of the part connection
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    # norm is the length of the connection
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue

                    # normalize the vec still (col, row)
                    vec = np.divide(vec, norm)

                    # generate midnum points in a list [(x, y)] ranging from -> to
                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    # get the x vectors from the paf_avg along the midpoints from -> to
                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])

                    # get the y vectors from the paf_avg along the midpoints from -> to
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    # get the dot products for each of the interpolated paf_avg points and the normalized vector from -> to (col, row)
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    
                    # float score taking into acount the size of the image and the distance between the points
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)

                    # at least .8 of the midpts have a dot product greater than thre2
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['dp_perc'] * len(
                        score_midpts)

                    # average dot product scaled by distance is greater than 0
                    criterion2 = score_with_dist_prior > 0

                    # if both criterion are good
                    if criterion1 and criterion2:
                        # add a list to the connection_candidate of the a index, b index, score with dist prior and
                        # the score summed with the heatmap scores
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])


            # connection_candidate is a list of lists for connection k with [[a index, b index, score_with_dist_prior, score_width_dist_prior + heatmapscores for a and b]]
            
            # sort the connection_candidates from high to low by their score_with_dist_prior
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3] # candA index, candB index, score_width_dist_prior
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    # append from peakID, toPeakId, score_width_dist_prior, candA index, candB index
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])



    print(f'missing candidates for {[[config.parts[config.limbs_conn[mapIdx[x][0] // 2][0]], config.parts[config.limbs_conn[mapIdx[x][0] // 2][1]]] for x in special_k]}')

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, numParts + 2))

    # all_peaks is a list of lists [(col, row, score, id), (col, row, score, id)] of peaks and scores and unique ids
    # candidate is an array of all the peaks (col, row, score, id)
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            # from peak id's
            partAs = connection_all[k][:, 0]
            # to peak ids
            partBs = connection_all[k][:, 1]
            # index into parts of from and to
            indexA, indexB = np.array(limbSeq[k])

            # iterate over possible connections
            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < numParts:#17:
                    row = -1 * np.ones(numParts + 2)#np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < params['min_num'] or subset[i][-2] / subset[i][-1] < params['thre3']:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = cv2.imread(input_image)  # B,G,R order
    for i in range(numParts):
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    stickwidth = 4

    for i in range(numLimbConn): #range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                       360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def visualize_paf(oriImg, paf_avg, idxFromMap, config):
    # # flip x coordinate in paf_avg #TODO why
    U = paf_avg[:,:,idxFromMap[0]] * -1
    # get y coordinate in paf_avg
    V = paf_avg[:,:,idxFromMap[1]]

    # generate x y coordinates of ogImg size
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

    # render the original image
    plt.figure()
    plt.imshow(oriImg[:,:,[2,1,0]], alpha = .5)

    # plot arrows in U and V at the coordinates of X and Y
    s = 5 # the stride used to plot arrows (5 is every fifth arrow)
    plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
                scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

    # generate a false array of ogImg size
    M = np.zeros(U.shape, dtype='bool')

    # mask vectors where the length of the paf vector is less than .5
    M[U**2 + V**2 < 0.5 * 0.5] = True

    # mask small vectors in paf
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
                scale=50, headaxislength=4, alpha=.5, width=0.001, color='b')

    limb = config.limbs_conn[idxFromMap[0] // 2]
    plt.title(f"paf from {config.parts[limb[0]]} to {config.parts[limb[1]]}")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.show()

def save_all_paf(oriImg, paf_avg, mapIdx, config):
    fig = plt.figure()
    fig.set_size_inches(40, 40)

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i, idxFromMap in enumerate(mapIdx):
        plt.subplot(len(mapIdx) // 3  + 1, 3, i + 1)

        # # flip x coordinate in paf_avg #TODO why
        U = paf_avg[:,:,idxFromMap[0]] * -1
        # get y coordinate in paf_avg
        V = paf_avg[:,:,idxFromMap[1]]

        # generate x y coordinates of ogImg size
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

        # render the original image
        plt.imshow(oriImg[:,:,[2,1,0]], alpha = .5)

        # plot arrows in U and V at the coordinates of X and Y
        s = 5 # the stride used to plot arrows (5 is every fifth arrow)
        plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
                    scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

        # generate a false array of ogImg size
        M = np.zeros(U.shape, dtype='bool')

        # mask vectors where the length of the paf vector is less than .5
        M[U**2 + V**2 < 0.5 * 0.5] = True

        # mask small vectors in paf
        U = ma.masked_array(U, mask=M)
        V = ma.masked_array(V, mask=M)

        plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s], 
                    scale=50, headaxislength=4, alpha=.5, width=0.001, color='b')

        limb = config.limbs_conn[idxFromMap[0] // 2]
        plt.title(f"paf from {config.parts[limb[0]]} to {config.parts[limb[1]]}")

    plt.savefig('paf_result.png')

def visualize_heat(oriImg, heatmap_avg, partIdx, config):

    # render the original image
    plt.figure()
    plt.imshow(oriImg[:,:,[2,1,0]])
    plt.imshow(heatmap_avg[:, :, partIdx], alpha = .5)

    plt.title(f"heatmap for {config.parts[partIdx]}")

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.show()

def save_all_heat(oriImg, heatmap_avg, numParts, config):
    fig = plt.figure()
    fig.set_size_inches(40, 40)

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for partIdx in range(numParts):
        plt.subplot(numParts // 3, 3, partIdx + 1)
        plt.imshow(oriImg[:,:,[2,1,0]])
        plt.imshow(heatmap_avg[:, :, partIdx], alpha = .5)

        plt.title(f"heatmap for {config.parts[partIdx]}")

    plt.savefig('heatmap_result.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')
    parser.add_argument('--config', type=str, default='NYU_Small_Hand', help='config from config.py')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    input_image = args.image
    output = args.output
    keras_weights_file = args.model
    config = GetConfig(args.config)


    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model(config.paf_layers, config.num_parts_with_background)
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    print('start processing...')
    tic = time.time()
    # generate image with body parts
    canvas = process(input_image, params, model_params, config)

    toc = time.time()
    print ('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

    cv2.destroyAllWindows()

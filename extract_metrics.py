import numpy as np
import matplotlib.pyplot as plt
import pickle
def entropy( probs ):

    E = np.sum( np.multiply( probs, np.log(probs+np.finfo(np.float32).eps) ) , axis=-1) / np.log(1.0/probs.shape[-1])
    return np.asarray( E, dtype="float32" )

def probdist( probs ):
    arrayA = np.asarray(np.argsort(probs,axis=-1), dtype="uint8")
    arrayD = np.ones( probs.shape[:-1], dtype="float32" )

    P = probs
    D = arrayD
    A = arrayA

    for i in range( arrayD.shape[0] ):
        for j in range( arrayD.shape[1] ):
            D[i,j] = ( 1 - P[ i, j, A[i,j,-1] ] + P[ i, j, A[i,j,-2] ] )

    return arrayD

def prediction(probs, gt, ignore=True ):
    pred = np.asarray( np.argmax( probs, axis=-1 ), dtype="uint8" )
    if ignore == True:
        pred[ gt==255 ] = 255
    return pred

def computeMetrics(gt, probs, seg, compo):
    flag = np.zeros( (gt.shape[0], gt.shape[1]), dtype="uint8" )
    #print(flag.shape)
    marked = np.zeros( (gt.shape[0], gt.shape[1]), dtype="uint8" )
    #print(marked.shape)
    #E:每个像素点的交叉熵(10,10)， D:probability margin(10, 10)
    heatmaps = { "E": entropy( probs ), "D": probdist( probs )}  
    #print(heatmaps['E'].shape, heatmaps['D'].shape)

    metrics = { "iou": list([]), "iou0": list([]), "class": list([]) } #, "mean_x": list([]), "mean_y": list([]) }

    #E和D和S的五种值，原始，in， bd， 相关， 相关in
    for m in list(heatmaps)+["S"]:
        metrics[m          ] = list([])
        metrics[m+"_in"    ] = list([])
        metrics[m+"_bd"    ] = list([])
        metrics[m+"_rel"   ] = list([])
        metrics[m+"_rel_in"] = list([])

    nclasses  = probs.shape[-1]
    #每个像素点的类别的拆分  
    for i in range(nclasses):
        metrics['cprob'+str(i)] = list([])
    #print(metrics)
    step = 8
    block = 0
    for i in range(0, probs.shape[0], step):
        for j in range(0, probs.shape[1], step):
    #         if flag[i, j] == 1:
    #             continue
            ssum = 0
            new_ii, new_jj = i, j
            while ssum <  step * step:
                for m in metrics:
                    metrics[m].append( 0 )
                #print(ssum)
                new_i, new_j = new_ii, new_jj
                #print(new_i, new_j)
                I, U = 0, 0 
                c = seg[new_i, new_j]
                block = block + 1
                ind = abs(compo[new_i, new_j])
                n_in, n_bd = 0, 0
                first = True
                #print(i + step,probs.shape[0], min(i + step, probs.shape[0]))
                for x in range(new_i, min(i + step, probs.shape[0])):
                    for  y in range(j, min(j + step, probs.shape[1])):
                        if flag[x, y] == 1:
                            continue
                        #print(x , y)
                        # compute all metrics
                        #print("in compute:n_in, n_bd", n_in, "^^^^^", n_bd)
                        if compo[x, y] in [ind, -ind]:
                            flag[x, y] = 1
                            ssum += 1
                            marked[x, y] = block
                            if compo[x,y] == ind:
                                for h in heatmaps:
        #                             print(heatmaps[h][x,y])
        #                             print(metrics[h+"_in"][-1])
                                    metrics[h+"_in"][-1] += heatmaps[h][x,y]
                                n_in += 1
                            elif compo[x,y] == -ind:
                                for h in heatmaps:
                                    metrics[h+"_bd"][-1] += heatmaps[h][x,y]
                                n_bd += 1
                            for ic in range(nclasses):
                                metrics["cprob"+str(ic)][-1] += probs[x,y,ic]

                            if gt != []:
                                if gt[x,y] == c:
                                    I += 1
                            U += 1
                        elif first:
                            new_ii = x
                            new_jj = y
                            first = False

                        metrics["class"   ][-1] = c
                        if gt != []:
                            metrics["iou"     ][-1] = float(I) / (float(U))
                            metrics["iou0"    ][-1] = int(I == 0)
                        else:
                            metrics["iou"     ][-1] = -1
                            metrics["iou0"    ][-1] = -1
                        metrics["S"       ][-1] = n_in + n_bd
                        metrics["S_in"    ][-1] = n_in
                        metrics["S_bd"    ][-1] = n_bd
                        if n_bd > 0:
                            metrics["S_rel"   ][-1] = float( n_in + n_bd ) / (float(n_bd))
                            metrics["S_rel_in"][-1] = float( n_in ) / (float(n_bd))
                        else:
                            metrics["S_rel"   ][-1] = float( n_in + n_bd ) / (float(n_bd) + step * 4)
                            metrics["S_rel_in"][-1] = float( n_in ) / (float(n_bd) + step * 4)
                        #metrics["mean_x"][-1] /= ( n_in + n_bd )
                        #metrics["mean_y"][-1] /= ( n_in + n_bd )

                        for nc in range(nclasses):
                            metrics["cprob"+str(nc)][-1] /= ( n_in + n_bd )

                        for h in heatmaps:
                            metrics[h          ][-1] = (metrics[h+"_in"][-1] + metrics[h+"_bd"][-1]) / (float( n_in + n_bd ))
                            if ( n_in > 0 ):
                                metrics[      h+"_in"][-1] /= (float(n_in))
                            if n_bd > 0:
                                metrics[h+"_bd"    ][-1] /= (float(n_bd))

                            metrics[h+"_rel"   ][-1] = metrics[h      ][-1] * metrics["S_rel"   ][-1]
                            metrics[h+"_rel_in"][-1] = metrics[h+"_in"][-1] * metrics["S_rel_in"][-1]
    return marked, metrics, block
    # print(marked[240:256, 128:144])
    # print(block)
    #     print(metrics)
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

def parse_rec(filename):#这个代码负责解析xml里面的标签，主要就是读取xml中关键内容，然后保存下来。
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects
 
def voc_ap(rec, prec, use_07_metric=False):#计算AP的函数，、
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
    # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
 

def voc_eval(detpath,
            annopath,
            imagesetfile,
            classname,
            cachedir,
            ovthresh=0.5,
            use_07_metric=False,
            use_diff=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    (default False)
    """
    # assumes detections are in detpath.format(classname)  
    # assumes annotations are in annopath.format(imagename) 
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file   
    
    # first load gt  读取真实标签
    if not os.path.isdir(cachedir): #判断缓存文件是否存在
        os.mkdir(cachedir)#不存在则创建一个
    cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)#创建一个缓存文件路径
    # read list of images
    with open(imagesetfile, 'r') as f:#打开imagesetfile
        lines = f.readlines()#直接全部读取
    imagenames = [x.strip() for x in lines]#去掉每个元素头和尾巴的字符
    print('cachefile', cachefile)
    if not os.path.isfile(cachefile):#如果缓存路径对应的文件没有，则载入读取annotations load annotations
        recs = {}#生成一个字典
        for i, imagename in enumerate(imagenames): #对于每一张图像进行循环
            recs[imagename] = parse_rec(annopath.format(imagename))#在字典里面放入每个图像缓存的标签路径
            if i % 100 == 0:#在这里输出标签读取进度。
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))#从这里可以看出来imagenames是什么，是一个测试集合的名字列表，这个Print输出进度。
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))#读取的标签保存到一个文件里面
        with open(cachefile, 'wb+') as f:#打开缓存文件
            pickle.dump(recs, f)#dump是序列化保存，load是反序列化解析
    else:#如果已经有缓存标签文件了，就直接读取
        # load
        with open(cachefile, 'rb') as f:
            try:
                recs = pickle.load(f)#用Load读取pickle里的文件
            except:
                recs = pickle.load(f, encoding='bytes')#如果读取不了，先二进制解码后再读取
    # extract gt objects for this class#从读取的换从文件中提取出一类的gt
    class_recs = {}
    npos = 0
    for imagename in imagenames:#存在Pickle里的是recs不是Imagenames，读取出来的也是recs
        R = [obj for obj in recs[imagename] if obj['name'] == classname]#首先用循环读取recs里面每一个图像名称里的目标类，从if obj['name']看出来Obj的类型是字典
        #recs里面是什么呢{图像名字：[‘第一个标签类别名字’：xxx，‘bbox’：[xxx,xxx,xxx,xxx]],   [‘第一个标签类别名字’：xxx，‘bbox’：[xxx,xxx,xxx,xxx]]，。。。。后面的都一样，有多少个标签就有多少个键值      }
        bbox = np.array([x['bbox'] for x in R])#R就是读取出来的该类别的键值对的值，Np.array就是一个指针指向一个多维数据，而不是多个指针指向每一个数据。节约内存左右啦
        if use_diff:#use_diff是之前设置的一个参数把，在R里面有一个键值对是'difficult':0，估计是一个多余操作，在目标检测里面没有这个设置
            difficult = np.array([False for x in R]).astype(np.bool)#astype是修改数据类型的，type是获取数据类型，
        else:
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)#反正就是改成0或者1的布尔数据类型呗，就是true或者false呗。
        det = [False] * len(R)#det我怀疑是detection的缩写，至于里面存放什么是个问号。
        npos = npos + sum(~difficult)#npos是之间设置为0的一个记录量，这里对difficult求和？可是我数据中这个都是0啊，就忽略把，可能是困难样本标记？反正关于dif的到这里也结束了，就是记录了一下
        class_recs[imagename] = {'bbox': bbox,#重点在这里，我要读取的是该类的所有gt，设置了一个量class_recs[]，上面都是提取操作，这里建立的class_是一个字典，键值对是图像名字：{框位置：，diffcult",det：}键值也是一个字典，总之就是二层字典。
                                'difficult': difficult,
                                'det': det}
 
    # read dets#det和dets是什么？从代码里面用print测试一下发现：det是与difficult相关的一个量，无所谓，但dets是检测结果的路径，读取出来就是图片名字、得分、bbox四个值来回。
    detfile = detpath.format(classname)
    print('detfile', detfile)
    with open(detfile, 'r') as f:#打开该类别的检测结果的txt
        lines = f.readlines()#直接读取全部
 
    splitlines = [x.strip().split(' ') for x in lines]#去掉\n
    image_ids = [x[0] for x in splitlines]#以图片名称为索引建立一个索引列表image_ids
    confidence = np.array([float(x[1]) for x in splitlines])#把第二列的得分提取出来作为一个列表，用np.array保存为连续取值，用一个指针解决
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])#对bbox进行数值类型转换，转换为float，这个代码真的很精炼...膜拜。
   
    nd = len(image_ids)#统计检测出来的目标数量
    tp = np.zeros(nd)#tp = true positive 就是检测结果中检测对的-检测是A类，结果是A类
    fp = np.zeros(nd)#fp = false positive 检测结果中检测错的-检测是A类，结果gt是B类。
 
    if BB.shape[0] > 0:#。shape是numpy里面的函数，用于计算矩阵的行数shape[0]和列数shape[1]
        # sort by confidence#按得分排序
        sorted_ind = np.argsort(-confidence)#如果confidence没有负号，就是从小到大排序，加了一个符号，直接改变功能，666.
        #总之np.argsort就是将得分从大到小排序，然后提取其排序结果对应原来的数据的索引，并输出到sorted_ind中，如sorted_ind第一个数是7，则代表原来第7个经过排序后再第一位。
        sorted_scores = np.sort(-confidence)#加个负号，从大到小排序，为什么和上面功能重复了？其实不重复，上面输出的是排序的索引，是一个索引，这里输出的是重新排序后数值结果。
        #其实sorted_ind才是用到的，下面也没有再用sorted_scores了，如果想自己看一看还可以用用。
        BB = BB[sorted_ind, :]#按排序的索引提取出数据放到BB里面。
        image_ids = [image_ids[x] for x in sorted_ind]#因为图像名称列表和得分列表出自同一个列表，得到排序索引之后，就可以按这个排序去提取Bbox了，
    
        # go down dets and mark TPs and FPs#标记正负样本
        for d in range(nd):#对于每一个检测结果
            R = class_recs[image_ids[d]]#首先用image_ids[]提取名称，作为键值对的键，去提取R,R就是类别名字、得分、坐标
            bb = BB[d, :].astype(float)#BB是置信度排序后的数据，bb就是把第d个元素的置信度从BB里面提取出来。
            ovmax = -np.inf#设置一个负无穷？
            BBGT = R['bbox'].astype(float)#提取真实的BB坐标，并且转换为跟检测结果一样的float变量
 
            if BBGT.size > 0:#size就是计算这个地方有没有gt，如果有，就计算交并比，那如果没有呢？？？？？？？？也就是说这个样本的计算值在分母中而不在分子中了？那就是只会拉低检测率。
                # compute overlaps
                # intersection#计算交并比的方法就是，对于左上角的(x,y)都取最大，右下角的坐标(x,y)都取最小，得到重叠区域。
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih#计算重叠区域面积
 
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +#计算交并比，计算来就是检测出的框面积+gt框面积，减掉重合的面积，就是总面积，然后除一下重叠面积。就是交、并比了。
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
 
                overlaps = inters / uni#这是交并比计算结果。
                ovmax = np.max(overlaps)#ovmax前面设为负无穷，所以这里其实就是如果有重叠区域，那么ovmax就是重叠区域，如果没有，就是负无穷。
                jmax = np.argmax(overlaps)#jmax也是计算最大，但是输出的是索引，与argsort差不多意思吧,反正Np里面有arg的可能都是输出索引。索引比实际数值可能更有用。
                #这里面有一个小点：overlaps不一定是一个取值，可能是一个列表。因为有些检测结果是多个Bbox无法区分，所以就会去对比找到交并比最大的，筛选检测结果。ovmax就是一个筛的过程
            if ovmax > ovthresh:#跟设置的阈值比较，如果大于，就是正样本。
                if not R['difficult'][jmax]:#如果不是difficule样本，就继续，否则打上负样本
                    if not R['det'][jmax]:#如果不是xxx，哎，反正跟这两层循环都没关系啦
                        tp[d] = 1.#就是满足阈值，打上正样本标签，
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.#不满足阈值打上负样本标签，加个点就是float了。
            else:
                fp[d] = 1.
 
    # compute precision recall#计算召回率
    fp = np.cumsum(fp)#计算负样本数量
    tp = np.cumsum(tp)#计算正样本数量
    rec = tp / float(npos)#计算召回率，给一个简单点的介绍，npos其实就是前面在gt中统计的样本数量啦，tp就是检测出来的样本
    #这里做一个简练的总结，recall就是在真实的样本与检测正确的真实样本的比，precision就是检测正确的样本与真实检测正确的样本的比。（很绕口，可以跳过。
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)#然后计算precision
    ap = voc_ap(rec, prec, use_07_metric)#得到recall和precision，调用voc_ap计算ap。
 
    return rec, prec, ap



from __future__ import print_function, division

import torch
from shutil import copyfile
from sys import exit
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
import torch.onnx as onnx
import onnx as onnx_origin
from onnxsim import simplify
import onnxruntime as rt
import configparser
import subprocess, sys
import glob

import cv2 

def list_files(directory, extension):
    return (f for f in sorted(os.listdir(directory)) if f.endswith('.' + extension))


def compute_f1(model_full, test_path, img_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = str(Path(test_path) / 'imgs')    

    strYTest = str(Path(test_path) / 'y_Test.txt')    

    with open(strYTest) as fInYTest:
        vecBMessYTest = fInYTest.readlines()
        vecBMessYTest = [bool(int(i)) for i in vecBMessYTest]        

    vecBMess = []    
    
    # specify ImageNet mean and standard deviation
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    

    str_imgs_list = list_files(path, "jpg")
    for str_img_name in str_imgs_list:
        str_img_name = str(Path(path) / str_img_name)
        
        img = cv2.imread(str_img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, img_size)       

        img = img.astype("float32") / 255.0
        img -= imagenet_mean
        img /= imagenet_std
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)        
        img = img.unsqueeze(0)
        img = img.to(device) 
        output = model_full.forward(img)
        
        val, cls = torch.max(output.data, 1)
        
        print(str_img_name) # added by Holy 2110250810
        print("[pytorch]--->predicted class:", cls.item())
        print("[pytorch]--->predicted value:", val.item())        

        preds = 1 - int(cls.item())
        vecBMess.append(bool(preds))
    
    vecBMessYTest_flip = [ not z for z in vecBMessYTest]
    vecBMess_flip = [ not z for z in vecBMess]

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    tp = sum(vecBResult)    

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest_flip, vecBMess)]
    fp = sum(vecBResult)    

    vecBResult = [ x and y for (x,y) in zip(vecBMessYTest, vecBMess_flip)]
    fn1 = sum(vecBResult)    

    prec = float(tp) / float(tp + fp)
    rec = float(tp) / float(tp + fn1)
    dF1 = 2 * prec * rec / (prec + rec)
    
    vecBResult = [ x == y for (x,y) in zip(vecBMessYTest, vecBMess)]    
    acc = float(sum(vecBResult)) / float(len(vecBResult))

    print(f'total frames: {len(vecBMess)}')
    total_frames_num = len(vecBMess)
    
    return dF1, tp, fp, fn1, prec, rec, acc, total_frames_num


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def copy_file(src_file, dst_file):
    try:
        copyfile(src_file, dst_file)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)


def update_file(file_path_name, src_str, dst_str):
    #read input file    
    with open(file_path_name, "rt") as fin:
        #read file contents to string
        data = fin.read()
        #replace all occurrences of the required string
        data = data.replace(src_str, dst_str)
    
    #open the input file in write mode    
    with open(file_path_name, "wt") as fin:
        #overrite the input file with the resulting data
        fin.write(data)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('automation_pytorch_config.ini')
    # first trained model
    model_full_path = 'model_ft_shufflenet_v2_0.pth'

    if not os.path.exists(model_full_path):
        files_path = os.path.dirname(model_full_path)
        if not files_path:
            files_path = '.'
        
        file_type = '\*' + os.path.splitext(model_full_path)[1][1:]
        pth_files = glob.glob(files_path + file_type)
        model_full_path = max(pth_files, key=os.path.getmtime)

    test_path = config['train']['test_dataset_path']

    img_size1 = config['train']['img_size'].split(',')

    img_size = (int(img_size1[0]), int(img_size1[1]))

    model_full = torch.load(model_full_path)

    dF1, tp, fp, fn1, prec, rec, acc, total_frames_num = compute_f1(model_full, test_path, img_size)

    model_full_path_first = model_full_path[:-4] + '_' + '{:.2f}'.format(dF1) + '.pth'

    copy_file(model_full_path, model_full_path_first)

    # second trained checkpoint
    pytorch_vision_version = 'pytorch/vision:' + config['train']['pytorch_vision_ver']
    shufflenet_version = 'shufflenet_' + config['train']['shufflenet_version']
    model_ft = torch.hub.load(pytorch_vision_version, shufflenet_version, pretrained=False) 
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()

    INIT_LR = 1e-3

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=INIT_LR)
    
    best_model_path = './best_model/best_model.pt'

    if not os.path.exists(best_model_path):
        files_path1 = os.path.dirname(best_model_path)
        if not files_path1:
            files_path1 = '.'
        
        file_type1 = '\*' + os.path.splitext(best_model_path)[1][1:]
        pth_files1 = glob.glob(files_path1 + file_type1)
        best_model_path = max(pth_files1, key=os.path.getmtime)
    
    # load the saved checkpoint
    model_ft, optimizer_ft, start_epochs, valid_loss_min = load_ckp(best_model_path, model_ft, optimizer_ft)

    model_ft.eval()

    dF1_2, tp_2, fp_2, fn1_2, prec_2, rec_2, acc_2, total_frames_num = compute_f1(model_ft, test_path, img_size)

    model_full_path_second = model_full_path[:-4] + '_ckp_' + '{:.2f}'.format(dF1_2) + '.pth'
    torch.save(model_ft, model_full_path_second)

    if dF1 > dF1_2:
        choosed_model_path = model_full_path_first
    else:
        choosed_model_path = model_full_path_second
    
    # added by Holy 2110280810
    latest_best_model_path = 'latest_best_shufflenet_model.pth'
    # copy_file(choosed_model_path, latest_best_model_path)
    # added by Holy 2111300810
    latest_best_model_path_tag = 'latest_best_shufflenet_model_' + config['train']['model_tag'] + '.pth'
    fine_tune = False
    if fine_tune:
        copy_file(choosed_model_path, latest_best_model_path_tag)
    else:
        copy_file(choosed_model_path, latest_best_model_path)
    # end of addition 2111300810
    # end of addition 2110280810

    print(f'first: f1:{dF1}, tp:{tp}, fp:{fp}, fn:{fn1}, prec:{prec}, rec:{rec}, acc:{acc}')
    print(f'second: f1:{dF1_2}, tp:{tp_2}, fp:{fp_2}, fn:{fn1_2}, prec:{prec_2}, rec:{rec_2}, acc:{acc_2}')

    # added by Holy 2109300810
    first_str = f'first: f1:{dF1}, tp:{tp}, fp:{fp}, fn:{fn1}, prec:{prec}, rec:{rec}, acc:{acc}'
    second_str = f'second: f1:{dF1_2}, tp:{tp_2}, fp:{fp_2}, fn:{fn1_2}, prec:{prec_2}, rec:{rec_2}, acc:{acc_2}'
    # end of addition 2109300810
    
    # export to simplified onnx
    model_ft_full = torch.load(choosed_model_path)
    
    # export to onnx
    input_image = torch.zeros((1,3,img_size[0],img_size[1]))
    input_image = input_image.to(device)
    
    input_names = ["x"]
    output_names = ["y"]

    #convert pytorch to onnx
    onnx_model_pathname = choosed_model_path[:-4] + '.onnx'
    torch_out = onnx.export(
        model_ft_full, input_image, onnx_model_pathname, input_names=input_names, output_names=output_names)
    
    # load your predefined ONNX model
    model = onnx_origin.load(onnx_model_pathname)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    # Save the ONNX model
    onnx_model_simplified_pathname = onnx_model_pathname[:-5] + '_simplified' + '.onnx'
    onnx_origin.save(model_simp, onnx_model_simplified_pathname)

    # verify pytorch onnx
    #test image
    img_path = os.path.join(test_path, 'imgs/img00001.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img1 = img.to(device)    

    #pytorch test
    model = torch.load(choosed_model_path)
    
    output = model.forward(img1)
    val, cls = torch.max(output.data, 1)
    print("[pytorch]--->predicted class:", cls.item())
    print("[pytorch]--->predicted value:", val.item())

    #onnx test
    sess = rt.InferenceSession(onnx_model_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    cls = np.argmax(output[0][0], axis=0)
    val = output[0][0][cls]
    print("[onnx]--->predicted class:", cls)
    print("[onnx]--->predicted value:", val)
    
    #simplified onnx test
    sess = rt.InferenceSession(onnx_model_simplified_pathname)
    
    x = "x"
    y = ["y"]
    output = sess.run(y, {x: img.numpy()})
    cls = np.argmax(output[0][0], axis=0)
    val = output[0][0][cls]
    print("[onnx_simplified]--->predicted class:", cls)
    print("[onnx_simplified]--->predicted value:", val)

    # copy ps1 file
    src_ps1_file = './templates/onnx_to_ncnn_shufflenet_v2.ps1'
    dst_ps1_file = './onnx_to_ncnn_shufflenet_v2.ps1'
    copy_file(src_ps1_file, dst_ps1_file)

    # update ps1 file
    src_ps1_str = 'onnx2ncnn.exe'
    dst_ps1_str = os.path.join(config['train']['onnx2ncnn_path'], 'onnx2ncnn.exe').replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'model_shufflenet_v2_simplified.onnx'
    dst_ps1_str = onnx_model_simplified_pathname.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'shufflenet_v2_x1_0.param'
    dst_ps1_str = onnx_model_pathname[:-5] + '.param'
    dst_ps1_str = dst_ps1_str.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)
    param_filename = os.path.basename(dst_ps1_str)
    param_pathname = dst_ps1_str
    dst_pytorch_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                             'support/holyMess/'+param_filename).replace('\\', '/')

    src_ps1_str = 'shufflenet_v2_x1_0.bin'
    dst_ps1_str = onnx_model_pathname[:-5] + '.bin'
    dst_ps1_str = dst_ps1_str.replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)
    bin_filename = os.path.basename(dst_ps1_str)
    bin_pathname = dst_ps1_str
    dst_pytorch_bin_path_name = os.path.join(config['train']['cpp_project_path'],
                                             'support/holyMess/'+bin_filename).replace('\\', '/')

    # run ps1
    p = subprocess.Popen(['pwsh.exe', dst_ps1_file], stdout=sys.stdout)
    p.communicate()
    
    # update cpp project
    str_time = '202109221716'

    src_cnn_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                           'support/holyMess/holy_param.ini').replace('\\', '/')
    dst_cnn_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                           'support/holyMess/holy_param_'+str_time+'.ini').replace('\\', '/')

    src_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_'+config['train']['model_tag'] +
                                       '.ini').replace('\\', '/')
    src_param_path_name_origin = os.path.join(config['train']['cpp_project_path'],
                                       'param.ini').replace('\\', '/')
    dst_param_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_' + config['train']['model_tag'] +
                                       '_' + str_time + '.ini').replace('\\', '/')
    
    # added by Holy 2110140810
    src_param_tmp_path_name = os.path.join(config['train']['cpp_project_path'],
                                       'param_tmp.ini').replace('\\', '/')
    
    with open(src_param_tmp_path_name) as f_cnn_param_tmp:
        cnn_param_tmp_lines = f_cnn_param_tmp.readlines()
        cnn_param_tmp_lines[0] = 'AlgorithmNO:0' + '\n'
        cnn_param_tmp_lines[1] = 'windingType:' + \
            config['train']['model_tag'] + '\n'
    with open(src_param_tmp_path_name, 'w') as f_cnn_param_tmp_w:
        f_cnn_param_tmp_w.writelines(cnn_param_tmp_lines)
    # end of addition 2110140810
    
    if os.path.exists(src_cnn_param_path_name):
        copy_file(src_cnn_param_path_name, dst_cnn_param_path_name)
    
    with open(src_cnn_param_path_name) as f_cnn_param:
        cnn_param_lines = f_cnn_param.readlines()
        cnn_param_lines[1] = 'holy_net = ./support/holyMess/' + \
            bin_filename + '\n'
        cnn_param_lines[2] = 'holy_param = ./support/holyMess/' + \
            param_filename + '\n'
        cnn_param_lines[3] = 'netW = ' + img_size1[0] + '\n'
        cnn_param_lines[4] = 'netH =' + img_size1[1] + '\n'
    with open(src_cnn_param_path_name, 'w') as f_cnn_param_w:
        f_cnn_param_w.writelines(cnn_param_lines)
    
    if os.path.exists(param_pathname):
        copy_file(param_pathname, dst_pytorch_param_path_name)

    if os.path.exists(bin_pathname):
        copy_file(bin_pathname, dst_pytorch_bin_path_name)
    
    if os.path.exists(src_param_path_name):
        copy_file(src_param_path_name, dst_param_path_name)
    else:
        copy_file(src_param_path_name_origin, src_param_path_name)
    with open(src_param_path_name) as f_param:
        param_lines = f_param.readlines()        
        param_lines[4] = 'demo_video:' + \
            config['train']['demo_video'] + '\n'
        param_lines[5] = 'test_dataset_path:' + \
            config['train']['test_dataset_path'] + '/\n'        
    with open(src_param_path_name, 'w') as f_param_w:
        f_param_w.writelines(param_lines)
    
    print(f'model_full_path: {model_full_path}')
    print(f'best_model_path: {best_model_path}')

    # added by Holy 2109290810
    dst_pytorch_param_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                             'build/install/bin/support/holyMess/'+param_filename).replace('\\', '/')

    dst_pytorch_bin_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                             'build/install/bin/support/holyMess/'+bin_filename).replace('\\', '/')

    dst_cnn_param_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                           'build/install/bin/support/holyMess/holy_param.ini').replace('\\', '/')

    dst_param_tag_build = os.path.join(config['train']['cpp_project_path'],
                                       'build/install/bin/param_'+config['train']['model_tag'] +
                                       '.ini').replace('\\', '/')
    
    dst_cnn_param_tmp_path_name_build = os.path.join(config['train']['cpp_project_path'],
                                           'build/install/bin/param_tmp.ini').replace('\\', '/') # added by Holy 2110140810
    
    if os.path.exists(dst_cnn_param_path_name_build):
        copy_file(src_cnn_param_path_name, dst_cnn_param_path_name_build)
        copy_file(dst_pytorch_param_path_name, dst_pytorch_param_path_name_build)
        copy_file(dst_pytorch_bin_path_name, dst_pytorch_bin_path_name_build)
        copy_file(src_param_path_name, dst_param_tag_build)

        src_param_str = 'bool_debug:false'
        dst_param_str = 'bool_debug:true'
        update_file(dst_param_tag_build, src_param_str, dst_param_str)

        copy_file(src_param_tmp_path_name, dst_cnn_param_tmp_path_name_build) # added by Holy 2110140810
    # end of addition 2109290810

    # added by Holy 2109300810
    # copy ps1 file
    src_ps1_file = './templates/run_demo.ps1'
    dst_ps1_file = './run_demo.ps1'
    copy_file(src_ps1_file, dst_ps1_file)

    # update ps1 file
    src_ps1_str = 'd:/backup/project/windingMessRope_holy/build/install/bin'
    dst_ps1_str = os.path.join(config['train']['cpp_project_path'],
                               'build/install/bin').replace('\\', '/')
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'first string'
    dst_ps1_str = first_str
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'second string'
    dst_ps1_str = second_str
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    src_ps1_str = 'total_frames_num'
    dst_ps1_str = f'total_frames_num: {total_frames_num}'
    update_file(dst_ps1_file, src_ps1_str, dst_ps1_str)

    # run ps1
    p = subprocess.Popen(['pwsh.exe', dst_ps1_file], stdout=sys.stdout)
    p.communicate()
    # end of addition 2109300810

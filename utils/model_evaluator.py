import argparse
import numpy as np
import os
import pandas as pd
import scipy.ndimage
import sys
import time
import torch

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '..')

from datasets.chromatograms_dataset import NpyChromatogramsDataset
from datasets.samplers import LoadingSampler
from datasets.transforms import ToTensor
from models.temperature_scaler import AlignmentTemperatureScaler, TemperatureScaler
from train.collate_fns import PadChromatogramsFor1DCNN

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def calibrate(
    calibration_dataset,
    model,
    batch_size=32,
    device='cpu',
    alpha=0.25,
    gamma=2,
    template_dataset=None,
    template_batch_size=4):
    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PadChromatogramsFor1DCNN())

    if template_dataset:
        template_dataloader = DataLoader(
            template_dataset,
            batch_size=template_batch_size,
            shuffle=False,
            collate_fn=PadChromatogramsFor1DCNN())

        template_dataloader = iter(cycle(template_dataloader))
        temperature_scaler = AlignmentTemperatureScaler(model, device)
        temperature_scaler.set_temperature(
            calibration_dataloader, template_dataloader, alpha, gamma)
    else:
        temperature_scaler = TemperatureScaler(model, device)
        temperature_scaler.set_temperature(
            calibration_dataloader, alpha, gamma)

    return temperature_scaler

def create_output_array(
    dataset,
    model,
    batch_size=32,
    device='cpu',
    load_npy=False,
    npy_dir='.',
    npy_name='output_array',
    threshold=0.5,
    template_dataset=None,
    template_batch_size=4):
    output_array = []

    if load_npy:
        output_array = np.load(os.path.join(npy_dir, npy_name + '.npy'))

        return output_array
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PadChromatogramsFor1DCNN())

    if template_dataset:
        template_dataloader = DataLoader(
            template_dataset,
            batch_size=template_batch_size,
            shuffle=False,
            collate_fn=PadChromatogramsFor1DCNN())

        template_dataloader = iter(cycle(template_dataloader))

    for batch in dataloader:
        chromatograms, labels = batch

        if template_dataset:
            template, template_label = next(template_dataloader)

        chromatograms = torch.from_numpy(
            np.asarray(chromatograms)).float().to(device)

        if template_dataset:
            template = torch.from_numpy(
                np.asarray(template)).float().to(device)

            template_label = torch.from_numpy(
                np.asarray(template_label)).float().to(device)

            output = model(chromatograms, template, template_label)
        else:
            output = model(chromatograms)

        output_array.append(output.detach().to('cpu').numpy())

    output_array = np.vstack(output_array)

    if len(output_array.shape) > 2:
        output_array = output_array.reshape(output_array.shape[0], -1)

    np.save(os.path.join(npy_dir, npy_name), output_array)

    return output_array

def create_rpn_results_file(
    dataset,
    model,
    batch_size=32,
    device='cpu',
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results_rpn.csv'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == len(dataset)

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT',
                'Window Size'
            ]
        ]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    for batch in dataloader:
        chromatograms, labels = batch

        chromatograms = torch.from_numpy(
            np.asarray(chromatograms)).float().to(device)

        output = model(chromatograms)

        output_idx = 0
        for i in range(idx, idx + len(labels)):
            print(idx)

            row = chromatograms.iloc[i]

            left_width, right_width, score = output[output_idx, 0, 1:]

            model_bounding_boxes.append([
                    row['ID'],
                    row['Filename'],
                    row['BB Start'],
                    row['BB End'],
                    int(round(left_width)),
                    int(round(right_width)),
                    row['OSW Score'],
                    score,
                    row['Lib RT'],
                    row['Window Size']])
            
            output_idx+= 1
            idx+= 1

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

def create_results_file(
    output_array,
    threshold=0.5,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    npy_name='output_array',
    results_csv='evaluation_results.csv',
    idx=None):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    if idx:
        chromatograms = chromatograms.iloc[idx]

    assert len(chromatograms) == output_array.shape[0]

    has_external_score = 'External Score' in list(chromatograms.columns)

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'External Precursor ID',
                'External Library RT/RT IDX',
                'Window Size',
                'External Label Left IDX',
                'External Label Right IDX',
                'External Score'
                'Model Predicted Left IDX',
                'Model Predicted Right IDX',
                'Model Score',
                'High Quality'
            ]
        ]

    label_output_array = np.zeros((output_array.shape))
    binarized_output_array = np.where(output_array >= threshold, 1, 0)
    inverse_binarized_output_array = (1 - binarized_output_array)

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i]
        binarized_output = binarized_output_array[i]
        inverse_binarized_output = inverse_binarized_output_array[i]

        left_width, right_width = np.argmax(output), np.argmax(output)

        score = np.max(output)

        high_quality = 0

        gaps = scipy.ndimage.find_objects(
            scipy.ndimage.label(inverse_binarized_output_array[i])[0])

        for gap in gaps:
            gap = gap[0]
            gap_length = gap.stop - gap.start

            if gap_length < 3:
                binarized_output[gap.start:gap.stop] = 1

        regions_of_interest = scipy.ndimage.find_objects(
            scipy.ndimage.label(binarized_output)[0])

        if regions_of_interest:
            scores = [(np.mean(output[r]) + np.max(output[r])) / 2 
                for r 
                in regions_of_interest]
            best_region_idx = np.argmax(scores)
            best_region = regions_of_interest[best_region_idx][0]

            start_idx, end_idx = best_region.start, best_region.stop

            left_width, right_width = start_idx, end_idx

            score = scores[best_region_idx]

            # Find relative index of highest intensity value in region of
            # interest and add to start index to get absolute index
            max_idx = np.argmax(output[regions_of_interest[best_region_idx]])
            max_idx+= start_idx

            roi_length = end_idx - start_idx

            if (2 < roi_length < 60 and
                start_idx < max_idx < end_idx and
                'DECOY_' not in row['Filename']):
                label_output_array[i, start_idx:end_idx] = np.ones(
                    (end_idx - start_idx)
                )

                high_quality = 1

        external_score = None

        if has_external_score:
            external_score = row['External Score']

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['External Precursor ID'],
                row['External Library RT/RT IDX'],
                row['Window Size'],
                row['External Label Left IDX'],
                row['External Label Right IDX'],
                external_score,
                left_width,
                right_width,
                score,
                high_quality])

    np.save(os.path.join(out_dir, 'label_' + npy_name), label_output_array)

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

def create_stats_eval_file(
    output_array,
    num_points=3,
    data_dir='OpenSWATHAutoAnnotated',
    chromatograms_csv='chromatograms.csv',
    out_dir='.',
    results_csv='evaluation_results.csv'):
    chromatograms = pd.read_csv(os.path.join(
        data_dir, chromatograms_csv))

    assert len(chromatograms) == output_array.shape[0]

    model_bounding_boxes = \
        [
            [
                'ID',
                'Filename',
                'Label BBox Start',
                'Label BBox End',
                'Pred BBox Start',
                'Pred BBox End',
                'OSW Score',
                'Model Score',
                'Lib RT',
                'Window Size'
            ]
        ]

    for i in range(len(chromatograms)):
        print(i)

        row = chromatograms.iloc[i]

        output = output_array[i, :]

        largest_idx = np.argmax(output)

        left_width = largest_idx - (num_points // 2)
        right_width = largest_idx + (num_points // 2)

        model_bounding_boxes.append([
                row['ID'],
                row['Filename'],
                row['BB Start'],
                row['BB End'],
                left_width,
                right_width,
                row['OSW Score'],
                str(output[largest_idx]),
                row['Lib RT'],
                row['Window Size']])

    model_bounding_boxes = pd.DataFrame(model_bounding_boxes)

    model_bounding_boxes.to_csv(
        os.path.join(out_dir, results_csv),
        index=False,
        header=False)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', type=str, default='.')
    parser.add_argument(
        '-dataset', '--dataset', type=str, default='hroest_Strep_600s_175pts.npy')
    parser.add_argument(
        '-chromatograms_csv',
        '--chromatograms_csv',
        type=str,
        default='chromatograms.csv')
    parser.add_argument(
        '-num_features', '--num_features', type=int, default=497)
    parser.add_argument(
        '-model_pth', '--model_pth', type=str, default='model.pth')
    parser.add_argument(
        '-out_dir', '--out_dir', type=str, default='results')
    parser.add_argument(
        '-results_csv', '--results_csv', type=str, default='results.csv')
    parser.add_argument('-threshold', '--threshold', type=float, default=0.5)
    parser.add_argument('-device', '--device', type=str, default='cpu')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=32)
    parser.add_argument(
        '-load_npy', '--load_npy', action='store_true', default=False)
    parser.add_argument('-npy_dir', '--npy_dir', type=str, default='results')
    parser.add_argument(
        '-npy_name', '--npy_name', type=str, default='output_array')
    parser.add_argument('-mode', '--mode', type=str, default='inference')
    parser.add_argument('-num_points', '--num_points', type=int, default=3)
    parser.add_argument(
        '-template_chromatograms_csv',
        '--template_chromatograms_csv',
        type=str,
        default='template_chromatograms.csv')
    parser.add_argument(
        '-template_dataset',
        '--template_dataset',
        type=str,
        default='hroest_Strep_600s_175pts.npy')
    parser.add_argument(
        '-template_labels',
        '--template_labels',
        type=str,
        default='hroest_Strep_600s_175pts_labels.npy')
    parser.add_argument(
        '-template_idx',
        '--template_idx',
        type=str,
        default='template_idx.txt')
    parser.add_argument(
        '-template_batch_size',
        '--template_batch_size',
        type=int,
        default=4)
    parser.add_argument(
        '-calibrate', '--calibrate', action='store_true', default=False)
    parser.add_argument(
        '-calibration_chromatograms_csv',
        '--calibration_chromatograms_csv',
        type=str,
        default='calibration_chromatograms.csv')
    parser.add_argument(
        '-calibration_dataset',
        '--calibration_dataset',
        type=str,
        default='hroest_Strep_600s_175pts.npy')
    parser.add_argument(
        '-calibration_labels',
        '--calibration_labels',
        type=str,
        default='hroest_Strep_600s_175pts_labels.npy')
    parser.add_argument(
        '-calibration_idx',
        '--calibration_idx',
        type=str,
        default='calibration_idx.txt')
    parser.add_argument(
        '-calibration_loss_alpha',
        '--calibration_loss_alpha',
        type=float,
        default=0.25)
    parser.add_argument(
        '-calibration_loss_gamma',
        '--calibration_loss_gamma',
        type=int,
        default=2)
    args = parser.parse_args()

    print(args)

    dataset = NpyChromatogramsDataset(
        args.data_dir,
        args.chromatograms_csv,
        args.dataset,
        labels=None,
        weak_labels=None,
        load_weak_labels=False,
        memmap=False,
        num_features=args.num_features,
        transform=ToTensor())

    if args.mode == 'alignment':
        template_dataset = NpyChromatogramsDataset(
            args.data_dir,
            args.template_chromatograms_csv,
            args.template_dataset,
            labels=args.template_labels,
            weak_labels=None,
            load_weak_labels=False,
            memmap=False,
            num_features=args.num_features,
            transform=ToTensor())

        sampling_fn = LoadingSampler(
            root_path=args.data_dir,
            filenames=[args.template_idx],
            dt='int',
            shuffle=True
        )
        evaluation_template_idx = sampling_fn()[0]
        template_dataset = Subset(template_dataset, evaluation_template_idx)

    if args.calibrate:
        calibration_dataset = NpyChromatogramsDataset(
            args.data_dir,
            args.calibration_chromatograms_csv,
            args.calibration_dataset,
            labels=args.calibration_labels,
            weak_labels=None,
            load_weak_labels=False,
            memmap=False,
            num_features=args.num_features,
            transform=ToTensor())

        sampling_fn = LoadingSampler(
            root_path=args.data_dir,
            filenames=[args.calibration_idx],
            dt='int',
            shuffle=True
        )
        calibration_idx = sampling_fn()[0]
        calibration_dataset = Subset(calibration_dataset, calibration_idx)

    print('Data loaded in {0:0.1f} seconds'.format(time.time() - start))

    model = torch.load(args.model_pth, map_location=args.device)

    if hasattr(model, 'get_model'):
        model = model.get_model()
    
    if args.mode == 'rpn':
        model.mode = 'test'
        model.pre_nms_topN = 1
        model.post_nms_topN = 1

    model.eval()

    if args.mode == 'rpn':
        create_rpn_results_file(
            dataset,
            model,
            args.batch_size,
            args.device,
            args.data_dir,
            args.chromatograms_csv,
            args.out_dir,
            args.results_csv)
    else:
        if args.mode == 'alignment':
            if args.calibrate:
                model = calibrate(
                    calibration_dataset,
                    model,
                    batch_size=args.batch_size,
                    device=args.device,
                    alpha=args.calibration_loss_alpha,
                    gamma=args.calibration_loss_gamma,
                    template_dataset=template_dataset,
                    template_batch_size=args.template_batch_size)
            else:
                model.probs = True
            
            output_array = create_output_array(
                dataset,
                model,
                args.batch_size,
                args.device,
                args.load_npy,
                args.npy_dir,
                args.npy_name,
                args.threshold,
                template_dataset,
                args.template_batch_size)
        else:
            if args.calibrate:
                if hasattr(model, 'set_output_probs'):
                    model.set_output_probs(False)

                model = calibrate(
                    calibration_dataset,
                    model,
                    batch_size=args.batch_size,
                    device=args.device,
                    alpha=args.calibration_loss_alpha,
                    gamma=args.calibration_loss_gamma)
            else:
                if hasattr(model, 'set_output_probs'):
                    model.set_output_probs(True)

            output_array = create_output_array(
                dataset,
                model,
                args.batch_size,
                args.device,
                args.load_npy,
                args.npy_dir,
                args.npy_name,
                args.threshold)
        
        if args.calibrate:
            save_name = 'calibrated_' + args.model_pth.split('/')[-1]
            save_path = os.path.join(args.out_dir, save_name)
            torch.save(model, save_path)

        if args.mode in ['alignment', 'inference']:
            create_results_file(
                output_array,
                args.threshold,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.npy_name,
                args.results_csv)
        elif args.mode == 'stats':
            create_stats_eval_file(
                output_array,
                args.num_points,
                args.data_dir,
                args.chromatograms_csv,
                args.out_dir,
                args.results_csv)

    print('It took {0:0.1f} seconds'.format(time.time() - start))

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
from CustomAgeModel import CustomAgeModel
import torch
from tqdm import tqdm
import argparse
from AGECustomDataset import getDataloader


def instantModel(device):
    model = CustomAgeModel(device)
    model.to(device)

    ethnicity_criterion = torch.nn.CrossEntropyLoss()
    gender_criterion = torch.nn.BCELoss()
    age_criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    return model, optimizer, {
        'ethnicity': ethnicity_criterion,
        'gender': gender_criterion,
        'age': age_criterion}

def train_model(models, dataset_loader, options):
    model, optimizer, criterions = models
    best_loss = 10000000000
    for epoch in(range(options.epochs)):
        for phase in ['train', 'validation']:
            is_train = True if phase == 'train' else False
            model.train(is_train)
            phase_loss = 0
            phase_eth_loss = 0
            phase_age_loss = 0
            phase_gender_loss = 0
            phase_eth_acc = []
            phase_gender_acc = []

            targetIter = tqdm(dataset_loader[phase])
            for idx, datas in enumerate(targetIter):
                img, label = datas
                img = img.to(options.device)
                eth_label = label['ethnicity'].to(options.device)
                gender_label = label['gender'].to(options.device).to(torch.float32)
                age_label = label['age'].to(options.device).to(torch.float32)

                optimizer.zero_grad()

                forward_pass = model(img)

                eth_out = forward_pass['ethnicity']
                gender_out = forward_pass['gender'].squeeze(1)
                age_out = forward_pass['age'].squeeze(1)

                eth_loss = criterions['ethnicity'](eth_out, eth_label)
                gender_loss = criterions['gender'](gender_out, gender_label)
                age_loss = criterions['age'](age_out, age_label)

                total_loss = eth_loss + gender_loss + age_loss

                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()

                phase_loss += total_loss.item()
                phase_eth_loss += eth_loss.item()
                phase_gender_loss += gender_loss.item()
                phase_age_loss += age_loss.item()
                phase_gender_acc.append((gender_label == gender_out).sum().item() / gender_label.size(0)) # (16, 1, 48, 48)
                phase_eth_acc.append((eth_label == eth_out.argmax(1)).sum().item() / eth_label.size(0))
                
                targetIter.set_description("Epoch {}/{}, total loss : {:2f}, ethnicity loss : {:2f},"
                                           "gender loss : {:2f}, age loss : {:2f}, gender accuracy : {:2f}%"
                                           "ethnicity accuracy : {:2f}%".format(epoch, options.epochs,
                                                                            total_loss.item(), eth_loss.item(),
                                                                            gender_loss.item(), age_loss.item(),
                                                                            phase_eth_acc[-1], phase_eth_acc[-1]))
                targetIter.refresh()
            print()
            print("Epoch {}/{} @ phase {} completed with total loss : {:2f}, total ethnicity loss : {:2f},"
                  "total gender loss : {:2f}, total age loss : {:2f}, ethnicity accuracy : {:2f}%, gender accuracy : {:2f}%".format(
                epoch, options.epochs, phase, phase_loss, phase_eth_loss, phase_gender_loss, phase_age_loss,
                sum(phase_eth_acc) / len(phase_eth_acc), sum(phase_gender_acc) / len(phase_gender_acc)))

            if phase == 'validation':
                print()
                print("===================================================================================")
                print()

                if phase_loss < best_loss:
                    torch.save(model, os.path.join(options.save_dir, 'age_gender_best.pth'))
                    best_loss = phase_loss
    return model.state_dict()

def test_model(model, dataset, criterion, options):
    crit_eth, crit_gender, crit_age = criterion['ethnicity'], criterion['gender'], criterion['age']
    total_loss = 0
    total_eth_loss = 0
    total_gender_loss = 0
    total_age_loss = 0
    eth_accuracy = []
    gender_accuracy = []

    with torch.no_grad():
        targetIter = tqdm(dataset['test'])

        for idx, datas in enumerate(targetIter):
            img, label = datas
            img = img.to(options.device)
            eth_label = label['ethnicity'].to(options.device)
            gender_label = label['gender'].to(options.device).to(torch.float32)
            age_label = label['age'].to(options.device).to(torch.float32)

            forward_pass = model(img)
            eth_loss = crit_eth(forward_pass['ethnicity'], eth_label)
            gender_loss = crit_gender(forward_pass['gender'].squeeze(1), gender_label)
            age_loss = crit_age(forward_pass['age'].squeeze(1), age_label)
            sum_loss = eth_loss + gender_loss + age_loss

            total_loss += sum_loss.item()
            total_eth_loss += eth_loss.item()
            total_gender_loss += gender_loss.item()
            total_age_loss += age_loss.item()
            eth_accuracy.append((eth_label == forward_pass['ethnicity'].argmax(1)).sum().item() / eth_label.size(0))
            gender_accuracy.append((gender_label == forward_pass['gender']).sum().item() / gender_label.size(0))
    print("Testing end with results <> total loss : {:2f}, total ethnicity loss : {:2f}, total gender loss : {:2f},"
          " total age loss : {:2f}, ethnicity accuracy : {:2f}%, gender accuracy : {:2f}%".format(total_loss,
                                                                                               total_eth_loss,
                                                                                               total_gender_loss,
                                                                                               total_age_loss,
                                                                                               sum(eth_accuracy)/len(eth_accuracy),
                                                                                               sum(gender_accuracy)/len(gender_accuracy)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, help='the directory to save pytorch model')
    parser.add_argument("--dataset_path", required=True, help='the root directory that store datas')
    parser.add_argument("--epochs", type=int, default=100, help='how many epoch for training')
    parser.add_argument("--batch_size", type=int, default=16, help='how many sample per forward')
    parser.add_argument("--workers", type=int, default=4, help='worker value for dataloader')
    parser.add_argument("--device", type=str, default='cuda:0', help='device id for training, cpu or cuda')

    opt = parser.parse_args()

    opt.train_dir = os.path.join(opt.dataset_path, 'train')
    opt.val_dir = os.path.join(opt.dataset_path, 'validation')
    opt.test_dir = os.path.join(opt.dataset_path, 'test')

    data_iterator = getDataloader(opt)
    models = instantModel(opt.device)

    model_statedict = train_model(models, data_iterator, opt)

    testModel = CustomAgeModel(opt.device)
    testModel.load_state_dict(model_statedict)
    testModel.eval()

    test_model(testModel, data_iterator, models[2], opt)


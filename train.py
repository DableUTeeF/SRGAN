import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import progress_bar, DotDict
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from models import Generator, Discriminator, FeatureExtractor

if __name__ == '__main__':

    opt = DotDict({})
    opt.add_argument('--dataset', value='folder')
    # opt.add_argument('--dataroot', value='/root/palm/PycharmProjects/DATA/SRGAN_HR/')
    opt.add_argument('--dataroot', value='/media/palm/Unimportant/DIV2K/')
    opt.add_argument('--workers', value=0)
    opt.add_argument('--batchSize', value=2)
    opt.add_argument('--imageSize', value=224)
    opt.add_argument('--upSampling', value=2)
    opt.add_argument('--nEpochs', value=100)
    opt.add_argument('--generatorLR', value=0.0001)
    opt.add_argument('--discriminatorLR', value=0.0001)
    opt.add_argument('--cuda', value=True)
    opt.add_argument('--nGPU', value=1)
    opt.add_argument('--generatorWeights', value='', )
    opt.add_argument('--discriminatorWeights', value='', )
    opt.add_argument('--out', value='checkpoints')

    try:
        os.makedirs(opt.out)
    except OSError:
        pass

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.RandomCrop(opt.imageSize * opt.upSampling),
                                    transforms.ToTensor()])

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    scale = transforms.Compose([transforms.ToPILImage(),
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])
                                ])

    dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    generator = Generator()
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))
    print(generator)

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
    print(discriminator)

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    print(feature_extractor)
    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    ones_const = Variable(torch.ones(opt.batchSize, 1))

    # if gpu is to be used
    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        ones_const = ones_const.cuda()

    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

    configure(
        'logs/' + opt.dataset + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR),
        flush_secs=5)

    low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    # Pre-train generator using raw MSE loss
    print('Generator pre-training')
    for epoch in range(2):
        mean_generator_content_loss = 0.0

        for i, data in enumerate(dataloader):
            # Generate data
            high_res_real, _ = data

            # Downsample images to low resolution
            for j in range(opt.batchSize):
                low_res[j] = scale(high_res_real[j])
                high_res_real[j] = normalize(high_res_real[j])

            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(high_res_real.cuda())
                high_res_fake = generator(Variable(low_res).cuda())
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))

            generator.zero_grad()

            generator_content_loss = content_criterion(high_res_fake, high_res_real)
            mean_generator_content_loss += generator_content_loss.data[0]

            generator_content_loss.backward()
            optim_generator.step()

            progress_bar(i, len(dataloader), f'Loss: {generator_content_loss.data[0]/(i+1):.{7}}')

        log_value('generator_mse_loss', mean_generator_content_loss / len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)

    # SRGAN training
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR * 0.1)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR * 0.1)

    print('SRGAN training')
    for epoch in range(opt.nEpochs):
        mean_generator_content_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        for i, data in enumerate(dataloader):
            # Generate data
            high_res_real, _ = data

            # Downsample images to low resolution
            for j in range(opt.batchSize):
                low_res[j] = scale(high_res_real[j])
                high_res_real[j] = normalize(high_res_real[j])

            # Generate real and fake inputs
            if opt.cuda:
                high_res_real = Variable(high_res_real.cuda())
                high_res_fake = generator(Variable(low_res).cuda())
                target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7).cuda()
                target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3).cuda()
            else:
                high_res_real = Variable(high_res_real)
                high_res_fake = generator(Variable(low_res))
                target_real = Variable(torch.rand(opt.batchSize, 1) * 0.5 + 0.7)
                target_fake = Variable(torch.rand(opt.batchSize, 1) * 0.3)

            ''' Train discriminator '''
            discriminator.zero_grad()

            discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                                 adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
            mean_discriminator_loss += discriminator_loss.data[0]

            discriminator_loss.backward()
            optim_discriminator.step()

            ''' Train generator '''
            generator.zero_grad()

            real_features = Variable(feature_extractor(high_res_real).data)
            fake_features = feature_extractor(high_res_fake)

            generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006 * content_criterion(
                fake_features, real_features)
            mean_generator_content_loss += generator_content_loss.data[0]
            generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
            mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

            generator_total_loss = generator_content_loss + 1e-3 * generator_adversarial_loss
            mean_generator_total_loss += generator_total_loss.data[0]

            generator_total_loss.backward()
            optim_generator.step()

            progress_bar(i, len(dataloader), f'DisLoss: {discriminator_loss.data[0]/(i+1):.{7}}, '
                                             f'GeneratorLoss(Content/Advers/Total): '
                                             f'{generator_content_loss.data[0]/(i+1):.{7}}/'
                                             f'{generator_adversarial_loss.data[0]/(i+1):.{7}}/'
                                             f'{generator_total_loss.data[0]/(i+1):.{7}}'
                         )

        # sys.stdout.write(
        #     '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (
        #         epoch, opt.nEpochs, i, len(dataloader),
        #         mean_discriminator_loss / len(dataloader), mean_generator_content_loss / len(dataloader),
        #         mean_generator_adversarial_loss / len(dataloader), mean_generator_total_loss / len(dataloader)))

        log_value('generator_content_loss', mean_generator_content_loss / len(dataloader), epoch)
        log_value('generator_adversarial_loss', mean_generator_adversarial_loss / len(dataloader), epoch)
        log_value('generator_total_loss', mean_generator_total_loss / len(dataloader), epoch)
        log_value('discriminator_loss', mean_discriminator_loss / len(dataloader), epoch)

        # Do checkpointing
        torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
        torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)

# Avoid closing
while True:
    pass

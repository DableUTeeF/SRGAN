from models import Generator
from PIL import Image
import numpy as np
import torch

if __name__ == '__main__':
    generator = Generator()
    generator.load_state_dict(torch.load('checkpoints/generator_final.pth'))
    generator = generator.cuda()

    testim = Image.open('/home/palm/Pictures/19601351_328276010932097_1689151741481236699_n.jpg')

    hi_im = testim.resize((448, 448))
    lo_im = testim.resize((224, 224))

    lo = np.array(lo_im)
    lo = np.rollaxis(lo, 2, 0)
    lo = lo.astype('float32')
    lo /= 127.5
    lo -= 1

    hi = generator(torch.cuda.FloatTensor(np.array([lo])))
    hi = hi.cpu().detach().numpy()
    hi = np.rollaxis(hi, 1, 4)
    hi = hi + 1
    hi *= 127.5
    hi = hi.astype('uint8')
    hi = np.clip(hi, 0, 255)

    fakehi = Image.fromarray(hi[0])

    hi_im.show()
    lo_im.resize((448, 448)).show()
    fakehi.show()

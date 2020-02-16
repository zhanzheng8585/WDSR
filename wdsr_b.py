
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, act=nn.ReLU(True), res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        # body.append(
        #     torch.nn.utils.weight_norm(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(
            torch.nn.utils.weight_norm(nn.Conv2d(n_feats, n_feats*expand, 1, padding=1//2)))
        body.append(act)
        body.append(
            torch.nn.utils.weight_norm(nn.Conv2d(n_feats*expand, int(n_feats*linear), 1, padding=1//2)))
        body.append(
            torch.nn.utils.weight_norm(nn.Conv2d(int(n_feats*linear), n_feats, kernel_size, padding=kernel_size//2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class MODEL(nn.Module):
    def __init__(self, args):
        super(MODEL, self).__init__()
        # hyper-params
        self.args = args
        scale = 2
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        # wn = lambda x: torch.nn.utils.weight_norm(x)

#       Batch's Size: batch_size x channel x H x W
        self.rgb_mean = torch.autograd.Variable(
            torch.FloatTensor([args.r_mean, args.g_mean, args.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            torch.nn.utils.weight_norm(nn.Conv2d(args.n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                Block(n_feats, kernel_size, act=act, res_scale=args.res_scale))

        # define tail module
        tail = []
        out_feats = scale*scale*args.n_colors
        tail.append(
            torch.nn.utils.weight_norm(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            torch.nn.utils.weight_norm(nn.Conv2d(args.n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        # copy variable to the display memory of GPU
        # x = (x - self.rgb_mean.cuda()*255)/127.5
        x = (x - self.rgb_mean*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        # x = x*127.5 + self.rgb_mean.cuda()*255
        x = x*127.5 + self.rgb_mean*255
        return x

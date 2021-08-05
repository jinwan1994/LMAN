from model import common
import torch
import torch.nn as nn

def make_model(args, parent=False):
    return SENET(args)

class SENET(nn.Module):
    def __init__(self, args, conv=common.default_conv, groups = 4):
        super(SENET, self).__init__()

        n_resgroups = args.n_resgroups
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [conv(n_feats, n_feats, kernel_size)]

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.resgroups = nn.ModuleList([common.ResGroup(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resgroups)])
        self.hse = nn.Sequential(common.HSAModule(n_feats))
        self.conv1 = nn.Sequential(conv(n_feats*n_resgroups, n_feats, 1))
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.n_resgroups = n_resgroups

    def forward(self, x):
        gs = [None] * self.n_resgroups
        x = self.sub_mean(x)
        x = self.head(x)
        for i in range(self.n_resgroups):
            if i == 0:
                gs[i] = self.resgroups[i](x)
            else:
                gs[i] = self.resgroups[i](gs[i-1])
        for j in range(self.n_resgroups):
            gs[j] = self.hse(gs[j])
        res = self.conv1(torch.cat(gs, 1))
        res = self.body(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

# if __name__ == "__main__":
    # net = SENET(upscale_factor=2)
    # total = sum([param.nelement() for param in net.parameters()])
    # print('   Number of params: %.2fM' % (total / 1e6))
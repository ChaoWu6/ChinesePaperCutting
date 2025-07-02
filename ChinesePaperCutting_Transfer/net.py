import torch.nn as nn
from model.configuration import TransModule_Config
from model.transformer_components import TransformerDecoderLayer
import torch
import numpy as np
import torchvision
########################################## VGG & components ##########################################

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


########################################## Transfer Module ##########################################

class TransModule(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self, config: TransModule_Config=None):
    super(TransModule, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(
          d_model=config.d_model,
          nhead=config.nhead,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          attn_drop=config.attn_drop,
          drop=config.drop,
          drop_path=config.drop_path,
          act_layer=config.act_layer,
          norm_layer=config.norm_layer,
          norm_first=config.norm_first
          ) \
      for i in range(config.nlayer)
    ])

  def forward(self, content_feature, style_feature):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature


# Example
# import torch
# transModule_config = TransModule_Config(
#             nlayer=3,
#             d_model=768,
#             nhead=8,
#             mlp_ratio=4,
#             qkv_bias=False,
#             attn_drop=0.,
#             drop=0.,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm,
#             norm_first=True
#             )
# transModule = TransModule(transModule_config)
# tgt = torch.randn(1, 20, 768)
# memory = torch.randn(1, 10, 768)
# print(transModule(tgt, memory).shape)


########################################## Decoder ##########################################

decoder_stem = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class Decoder_MVGG(nn.Module):
  def __init__(self, d_model=768, seq_input=False):
      super(Decoder_MVGG, self).__init__()
      self.d_model = d_model
      self.seq_input = seq_input
      self.decoder = nn.Sequential(
        # Proccess Layer 1        

        # Upsample Layer 2
        nn.ReflectionPad2d(1),
        nn.Conv2d(int(self.d_model), 256, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 256, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(256, 128, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 128, 3, 1, 0),
        nn.ReLU(),

        # Upsample Layer 4
        nn.ReflectionPad2d(1),
        nn.Conv2d(128, 64, 3, 1, 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 64, 3, 1, 0),
        nn.ReLU(),

        # Channel to 3
        nn.ReflectionPad2d(1),
        nn.Conv2d(64, 3, 3, 1, 0),
      )
        
        
  def forward(self, x, input_resolution):
    if self.seq_input == True:
      B, N, C = x.size()
#       H, W = math.ceil(self.img_H//self.patch_size), math.ceil(self.img_W//self.patch_size)
      (H, W) = input_resolution
      x = x.permute(0, 2, 1).reshape(B, C, H, W)
    x = self.decoder(x)  
    return x


# Example 1
# import torch
# decoder = Decoder_MVGG(d_model=768, seq_input=True)
# x = torch.randn(1, 3087, 768)
# y = decoder(x, input_resolution=(63, 49))
# print(y.shape)

from typing import Tuple, Union, Optional, List

########################################## Net ##########################################
class CutLoss():
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = int = 256
        self.patch_size: List[int] = [1, 2]

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        # bs, c,res,res = ref_noise.shape
        # #res = int(np.sqrt(res2))

        # ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2) 
        # trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)
        ref_noise_reshape = ref_noise
        trg_noise_reshape = trg_noise
        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss*0.06

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=2, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss
class Net(nn.Module):
  def __init__(self, encoder, decoder, transModule, lossNet):
    super(Net, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.encoder = encoder
    self.decoder = decoder
    self.transModule = transModule
    self.cut = CutLoss()
    # features of intermediate layers
    lossNet_layers = list(lossNet.children())
    self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
    self.feat_2 = nn.Sequential(*lossNet_layers[4:11]) # relu1_1 -> relu2_1
    self.feat_3 = nn.Sequential(*lossNet_layers[11:18]) # relu2_1 -> relu3_1
    self.feat_4 = nn.Sequential(*lossNet_layers[18:31]) # relu3_1 -> relu4_1
    self.feat_5 = nn.Sequential(*lossNet_layers[31:44]) # relu3_1 -> relu4_1

    # fix parameters
    for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
      for param in getattr(self, name).parameters():
        param.requires_grad = False


  # get intermediate features
  def get_interal_feature(self, input):
    result = []
    for i in range(5):
      input = getattr(self, 'feat_{:d}'.format(i+1))(input)
      result.append(input)
    return result
  

  def calc_content_loss(self, input, target, norm = False):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    if norm == False:
        return self.mse_loss(input, target) 
    else:
        return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))


  def calc_style_loss(self, input, target):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return self.mse_loss(input_mean, target_mean) + \
        self.mse_loss(input_std, target_std)


  # calculate losses
  def forward(self, i_c, i_s):
    f_c = self.encoder(i_c)
    f_s = self.encoder(i_s)
    f_c, f_c_reso = f_c[0], f_c[2]
    f_s, f_s_reso = f_s[0], f_s[2]
    f_cs = self.transModule(f_c, f_s)
    f_cc = self.transModule(f_c, f_c)
    f_ss = self.transModule(f_s, f_s)
    
    i_cs = self.decoder(f_cs, f_c_reso)
    i_cc = self.decoder(f_cc, f_c_reso)
    i_ss = self.decoder(f_ss, f_c_reso)
    


    f_c_loss = self.get_interal_feature(i_c)
    f_s_loss = self.get_interal_feature(i_s)
    f_i_cs_loss = self.get_interal_feature(i_cs)
    f_i_cc_loss = self.get_interal_feature(i_cc)
    f_i_ss_loss = self.get_interal_feature(i_ss)
    loss_cut = 0
    for i in range(4):
       
      l_cut = self.cut.get_attn_cut_loss(f_c_loss[i],f_i_cs_loss[i])
      loss_cut += l_cut


    loss_id_1 = self.mse_loss(i_cc, i_c) + self.mse_loss(i_ss, i_s)

    loss_c, loss_s, loss_id_2 = 0, 0, 0
    
    loss_c = self.calc_content_loss(f_i_cs_loss[-2], f_c_loss[-2], norm=True) + \
             self.calc_content_loss(f_i_cs_loss[-1], f_c_loss[-1], norm=True)
    for i in range(1, 5):
      loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
      loss_id_2 += self.mse_loss(f_i_cc_loss[i], f_c_loss[i]) + self.mse_loss(f_i_ss_loss[i], f_s_loss[i])
    
    return loss_c, loss_s, loss_id_1, loss_id_2, loss_cut,i_cs



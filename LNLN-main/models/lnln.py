import torch
from torch import nn
from .basic_layers import Transformer, CrossTransformer, HhyperLearningEncoder, GradientReversalLayer
from .bert import BertTextEncoder
from einops import rearrange, repeat


class LNLN(nn.Module):
    def __init__(self, args):
        super(LNLN, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_p = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][0], 
                        dim=args['model']['feature_extractor']['hidden_dims'][0], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][2], 
                        dim=args['model']['feature_extractor']['hidden_dims'][2], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][1], 
                        dim=args['model']['feature_extractor']['hidden_dims'][1], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        
        #对抗学习的生成器
        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'], 
            save_hidden=False, 
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'], 
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'], 
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'], 
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'], 
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])

        #GradientReversalLayer 是一种用于对抗性训练（Adversarial Training）的层，通常在多模态学习或域适应任务中使用。其主要作用是反转梯度，使得模型在训练过程中能够区分源域和目标域的数据。
        self.GRL = GradientReversalLayer(alpha=1.0)

        #对抗学习的鉴别器
        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            #self.completeness_check[0]
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'], 
                        save_hidden=False, 
                        token_len=args['model']['dmc']['completeness_check']['token_length'], 
                        dim=args['model']['dmc']['completeness_check']['input_dim'], 
                        depth=args['model']['dmc']['completeness_check']['depth'], 
                        heads=args['model']['dmc']['completeness_check']['heads'], 
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),
            #self.completeness_check[1]
            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'], int(args['model']['dmc']['completeness_check']['hidden_dim']/2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim']/2), 1),
                nn.Sigmoid()),
        ])


        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'], 
                        save_hidden=False, 
                        token_len=None, 
                        dim=args['model']['reconstructor']['input_dim'], 
                        depth=args['model']['reconstructor']['depth'], 
                        heads=args['model']['reconstructor']['heads'], 
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])


        self.dmml = nn.ModuleList([
            #self.dmml[0]
            Transformer(num_frames=args['model']['dmml']['language_encoder']['input_length'], 
                        save_hidden=True, 
                        token_len=None, 
                        dim=args['model']['dmml']['language_encoder']['input_dim'], 
                        depth=args['model']['dmml']['language_encoder']['depth'], 
                        heads=args['model']['dmml']['language_encoder']['heads'], 
                        mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim']),
            #self.dmml[1]
            HhyperLearningEncoder(dim=args['model']['dmml']['hyper_modality_learning']['input_dim'], 
                                  dim_head=int(args['model']['dmml']['hyper_modality_learning']['input_dim']/args['model']['dmml']['hyper_modality_learning']['heads']),
                                  depth=args['model']['dmml']['hyper_modality_learning']['depth'], 
                                  heads=args['model']['dmml']['hyper_modality_learning']['heads']),
            #self.dmml[2]
            CrossTransformer(source_num_frames=args['model']['dmml']['fuison_transformer']['source_length'], 
                             tgt_num_frames=args['model']['dmml']['fuison_transformer']['tgt_length'], 
                             dim=args['model']['dmml']['fuison_transformer']['input_dim'], 
                             depth=args['model']['dmml']['fuison_transformer']['depth'], 
                             heads=args['model']['dmml']['fuison_transformer']['heads'], 
                             mlp_dim=args['model']['dmml']['fuison_transformer']['hidden_dim']),
            #self.dmml[3]
            nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])
        ])



    def forward(self, complete_input, incomplete_input):
        #完整数据
        vision, audio, language = complete_input
        #缺失数据
        vision_m, audio_m, language_m = incomplete_input

        #batch_size
        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        self_bertmodel_language_m=self.bertmodel(language_m)
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        #print('self.bertmodel(language_m)',self.bertmodel(language_m).shape,self.bertmodel(language_m))
        #print('vision_m',vision_m.shape,vision_m)
        #print('audio_m',audio_m.shape,audio_m)



        #完备性检测器的输出
        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp) # completeness scores

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b = b)
        #生成器的输出
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1-w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)


        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)
        h_d_list = self.dmml[0](h_1_d)
        h_hyper = self.dmml[1](h_d_list, h_1_a, h_1_v, h_hyper)

        feat = self.dmml[2](h_hyper, h_d_list[-1])
        # Two ways to get the cls_output: using extra cls_token or using mean of all the features
        # output = self.dmml[3](feat[:, 0])
        output = self.dmml[3](torch.mean(feat[:, 1:], dim=1))

        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            # Reconstruction
            # for layer in self.reconstructor:
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            # Compute the complete features as the label of reconstruction 将完整的特征作为再现器的标签进行计算。
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            #检测器的输入，这里为啥是h_1_d啊，我看图里面是h_1_p ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            #对抗学习里，检测器的输入应该是真实样本和生成样本，应该是h_l_1,h_l_new吧
            effective_discriminator_input = rearrange(torch.cat([h_1_d, complete_language_feat]), 'b n d -> (b n) d')
            #检测器的输出
            effectiveness_discriminator_out = self.effective_discriminator(effective_discriminator_input)

            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1) # as the label of reconstruction

        #output是dmml模块的输出，w是完备性检测器的输出，effectiveness_discriminator_out是检测器的输出，rec_feats是再现器的输出结果（特征向量），complete_feats是以无人工设置缺失数据得到的特征向量
        return {'sentiment_preds': output,
                'w': w,
                'effectiveness_discriminator_out': effectiveness_discriminator_out,
                'rec_feats': rec_feats,
                'complete_feats': complete_feats}



def build_model(args):
    return LNLN(args)
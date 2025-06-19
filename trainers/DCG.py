import copy
import json
import os.path as osp
import random

import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from einops import rearrange, repeat
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

from einops import rearrange, repeat
from omegaconf import OmegaConf

import torch.nn.functional as F
from trainers.m3dloss import M3DLoss

_tokenizer = _Tokenizer()

dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "a photo of a {}, a type of texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg, design_details=None):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if design_details is None:
        design_details = {
            "trainer": "DCG",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": cfg.TRAINER.DCG.N_CTX,
            "maple_length_v":cfg.TRAINER.DCG.N_CTX_V,
        }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [
            x,
            compound_prompts_deeper_text,
            0,
        ]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
            x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
            @ self.text_projection
        )

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, clip_model_distill=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DCG.N_CTX
        n_ctx_vision = cfg.TRAINER.DCG.N_CTX_V
        #ctx_init = cfg.TRAINER.CoPrompt.CTX_INIT
        ctx_init_flag = cfg.TRAINER.DCG.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        M = cfg.TRAINER.DCG.M #the number of our visual prompts
        N = cfg.TRAINER.DCG.N
        self.MIX = cfg.TRAINER.DCG.MIX
        self.M = M
        self.N = N

        assert (
            cfg.TRAINER.DCG.PROMPT_DEPTH >= 1
        ), "For DCG, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = (
            cfg.TRAINER.DCG.PROMPT_DEPTH
        )  # max=12, but will create 11 such shared prompts
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        template_dict = {'Caltech101': ["a photo of a","this is a photo","this is picture of","one picture of a"], 
                         'DescribableTextures':['a photo of a texture', "this is a photo texture","this is a picture texture","one picture of a texture"],
                         'EuroSAT':['a centered satellite photo of', 'a centered satellite picture of','this is centered satellite photo of','one centered satellite photo of a'], 
                         'FGVCAircraft':['a photo of an aircraft','a picture of an aircraft','this is aircraft picture of','one picture of an aircraft'],
                         'Food101':['a photo of a food', 'this is a food photo', ' this is food picture of','one picture of a food'], 
                         'ImageNet':["a good photo of a X","this is a bad photo X","a good photo of a X","one picture of a X","a bad photo of the X","a photo of the large X"],
                         'OxfordFlowers':['a photo of a flower', 'one picture of a flower','this is flower picture of','one picture of a flower'],
                         'OxfordPets':['a photo of a pet', 'one picture of a pet','this is pet picture of','one picture of a pet'],
                         'StanfordCars':["a photo of a X X X X","this is a photo X X X X","this is picture of X X X X","one picture of a X X X X"],
                         'SUN397':["a photo of a","this is a photo","this is picture of","one picture of a"],
                         'UCF101':['a photo of a person doing', 'this is a photo people doing', 'this is picture of people doing', 'one picture of a person doing'],
                        'ImageNetV2':["a good photo of a X","this is a bad photo X","a good photo of a X","one picture of a X","a bad photo of the X","a photo of the large X"],
                        'ImageNetSketch':["a good photo of a X","this is a bad photo X","a good photo of a X","one picture of a X","a bad photo of the X","a photo of the large X"],
                        'ImageNetR':["a good photo of a X","this is a bad photo X","a good photo of a X","one picture of a X","a bad photo of the X","a photo of the large X"],
                        'ImageNetA':["a good photo of a X","this is a bad photo X","a good photo of a X","one picture of a X","a bad photo of the X","a photo of the large X"],}
        
        if self.MIX > min(n_ctx,n_ctx_vision):
            pass
        
        if ctx_init_flag is not None:
            # use given words to initialize context vectors
            ctx_list = template_dict[cfg.DATASET.NAME]
            n_word_ctx = len(ctx_list[0].split()) 
            ctx_vectors_list = []
            prompt_prefix_list = []
            for i in range(N):
                ctx_init = ctx_list[i].replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                    
                ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                prompt_prefix_list.append(prompt_prefix)
                
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            ctx_vectors = torch.stack(ctx_vectors_list)
            
            
            

        else:
            # random initialization
            ctx_vectors = torch.empty(N, n_ctx, ctx_dim, dtype=dtype)
            ctx_vision_vectors = torch.empty(M, n_ctx_vision ,768, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vision_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("DCG design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of DCG context words (tokens): {n_ctx}")
 
        self.proj = nn.Linear(ctx_dim, 768)
        if dtype == torch.float16:
            self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_vision = nn.Parameter(ctx_vision_vectors)
        
        self.compound_prompts_text = nn.ParameterList(
            [
                nn.Parameter(torch.empty(N, n_ctx, 512))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        
        self.visual_deep_prompts = nn.ParameterList(
            [
                nn.Parameter(torch.empty(M, n_ctx_vision, 768))
                for _ in range(self.compound_prompts_depth - 1)
            ]
        )
        
        
        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
            
        for single_para in self.visual_deep_prompts:
            nn.init.normal_(single_para, std=0.02)
        
        #single_layer = nn.Sequential(nn.Linear(ctx_dim, 768),nn.ReLU(),nn.Linear(768, 768))
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(
            single_layer, self.compound_prompts_depth - 1, min(self.M,self.N)
        ) 

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompt_list = []
        
        if ctx_init is not None:
            for i in range(N):
                prompt_prefix = prompt_prefix_list[i]
                prompts = [prompt_prefix + " " + name + "." for name in classnames] # 100
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) # 100x77
                prompt_list.append(tokenized_prompts)
            tokenized_prompts = torch.cat(prompt_list)
        else:
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            tokenized_prompts = tokenized_prompts.repeat(N,1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        

        clip_model_ = clip_model_distill
        if cfg.TRAINER.DCG.PREC == "fp32" or cfg.TRAINER.DCG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_.float()
        if torch.cuda.is_available():
            clip_model_.cuda()

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)


        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)
            
        ctx = ctx.permute(1, 0, 2, 3) #  N 100 16 512
        
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        

        return prompts

    def forward(self):
        ctx = self.ctx
        ctx_vision = self.ctx_vision
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
           
        
        if self.MIX>0:

            if self.M>self.N:
                ctx_vision_fix = self.proj(self.ctx[:self.N,:self.MIX])
                ctx_vision.data[:self.N,:self.MIX] = ctx_vision_fix.data
            else:
                ctx_vision_fix = self.proj(self.ctx[:self.M,:self.MIX])
                ctx_vision.data[:self.M,:self.MIX] = ctx_vision_fix.data
        
            for n_distribute,layers in enumerate(self.compound_prompt_projections):
                for index,layer in enumerate(layers):
                    self.visual_deep_prompts[index].data[n_distribute,:self.MIX] = layer(self.compound_prompts_text[index][n_distribute,:self.MIX]).data
 
        
        
        return (prompts, ctx_vision,self.compound_prompts_text,self.visual_deep_prompts)  # pass here original, as for visual 768 is required


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    
class CustomCLIP(nn.Module):
    def __init__(
        self, cfg, classnames, clip_model, clip_model_distill, clip_prompt_weights
    ):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(
            cfg, classnames, clip_model, clip_model_distill
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embedding = clip_prompt_weights
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
#         self.distill_criteria = cfg.TRAINER.DISTILL
        self.model_distill = clip_model_distill
        
        #self.lambd = cfg.TRAINER.W
        self.N = cfg.TRAINER.DCG.N
        self.M = cfg.TRAINER.DCG.M
        self.adapter_image = Adapter(512, 4).to(clip_model.dtype)
        self.adapter_text = Adapter(512, 4).to(clip_model.dtype)

        self.image_adapter_m = torch.tensor([0.05,0.1,0.15,0.15,0.15])[:self.M].cuda()
        self.text_adapter_m = torch.tensor([0.1,0.25,0.2,0.25,0.2])[:self.N].cuda()   
        
        
        self.m3dloss = M3DLoss("laplace")
        self.cfg = cfg
        self.n_cls = len(classnames)
        self.eps = 0.1
        self.max_iter = 100
        

        
        
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def forward(self, image1, image2=None, image3=None,image4=None,image5=None,label=None):
        b = image1.shape[0]
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        (
            prompts,
            vision_prompts,
            compound_prompts_text,
            visual_deep_prompts
        ) = self.prompt_learner()
        
        tokenized_prompts = self.tokenized_prompts
    
        text_features = self.text_encoder(
            prompts, tokenized_prompts,compound_prompts_text
        ).contiguous().view(self.N, self.n_cls, 512)#[76, 512]
        
        

        
        image_features = self.image_encoder(image1.type(self.dtype),vision_prompts,visual_deep_prompts).view(self.M, b , 512)


        x_a = self.adapter_image(image_features)
        image_features = (
            self.image_adapter_m[:,None,None] * x_a + (1 - self.image_adapter_m[:,None,None]) * image_features
        )
        # [4, 128, 512]
        x_b = self.adapter_text(text_features)
        text_features = (
            self.text_adapter_m[:,None,None] * x_b + (1 - self.text_adapter_m[:,None,None]) * text_features
        )
 
        image_features =  F.normalize(image_features, dim=2)  # N c d 
        text_features = F.normalize(text_features, dim=2)
 
        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous() 
        
        sim = sim.view(self.M,self.N,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim
        xx=torch.zeros(b*self.n_cls, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
        yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        
        
        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            
            return None
        
        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)
        logits = logit_scale * sim_op
       
       

        if self.prompt_learner.training:
            perm = torch.randperm(self.ori_embedding.shape[0])
            n_idx = perm[:self.N]
            pre_trained_text_features = self.ori_embedding[n_idx]
            
            pre_trained_image_features1 = self.model_distill.encode_image(image1)
        
            if image2 is not None:
                pre_trained_image_features2 = self.model_distill.encode_image(image2)
            else:
                pre_trained_image_features2 = self.model_distill.encode_image(image1)
            
            if image3 is not None:
                pre_trained_image_features3 = self.model_distill.encode_image(image3)
            else:
                pre_trained_image_features3 = self.model_distill.encode_image(image1)
                
            if image4 is not None:
                pre_trained_image_features4 = self.model_distill.encode_image(image4)
            else:
                pre_trained_image_features4 = self.model_distill.encode_image(image1)
                
            if image5 is not None:
                pre_trained_image_features5 = self.model_distill.encode_image(image5)
            else:
                pre_trained_image_features5 = self.model_distill.encode_image(image1)
                
            pre_train_image_features_list = torch.stack([pre_trained_image_features2,pre_trained_image_features3,pre_trained_image_features4,pre_trained_image_features5,pre_trained_image_features1])
            pre_trained_image_features = pre_train_image_features_list[:self.M]
            

            pre_trained_text_features = F.normalize(pre_trained_text_features, dim=2) 
            pre_trained_image_features = F.normalize(pre_trained_image_features, dim=2)

            
            
            
            if self.cfg.DATASET.SUBSAMPLE_CLASSES=="base": 
                

                loss = F.cross_entropy(logits, label)
                loss_distill_text = self.m3dloss(text_features.permute(1,0,2).float(),pre_trained_text_features.permute(1,0,2).float()).half()
                loss_distill_image = self.m3dloss(image_features.permute(1,0,2).float(),pre_trained_image_features.permute(1,0,2).float()).half()
                loss_distill =  2*loss_distill_text+0.1*loss_distill_image     
             
                  
                return loss + 0.15*loss_distill  
            else:
               
                teacher_sim = torch.einsum('mbd,ncd->mnbc', pre_trained_image_features, pre_trained_text_features).contiguous() 
                teacher_sim = teacher_sim.view(self.M,self.N,b*self.n_cls)
                teacher_sim = teacher_sim.permute(2,0,1)
                teacher_wdist = 1.0 - teacher_sim
                xx=torch.zeros(b*self.n_cls, self.M, dtype=sim.dtype, device=sim.device).fill_(1. / self.M)
                yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
                

                with torch.no_grad():
                    KK = torch.exp(-teacher_wdist.float() / self.eps)
                    T = self.Sinkhorn(KK,xx,yy)
                if torch.isnan(T).any():
                    return None
        
                sim_op = torch.sum(T * teacher_sim, dim=(1, 2))
                sim_op = sim_op.contiguous().view(b,self.n_cls)
                teacher_logits = logit_scale * sim_op
                L_ukd = F.kl_div(F.log_softmax(logits / 1, dim=1),F.softmax(teacher_logits / 1, dim=1),reduction='sum',) * (1 * 1) / logits.numel()
                loss_distill_text = self.m3dloss(text_features.permute(1,0,2).float(),pre_trained_text_features.permute(1,0,2).float())
                loss_distill_image = self.m3dloss(image_features.permute(1,0,2).float(),pre_trained_image_features.permute(1,0,2).float())

                loss_distill =  2*loss_distill_text+0.1*loss_distill_image   # 2 0.1  euro:200 0.1  0
                loss = self.cfg.TRAINER.DISTILL_W1*L_ukd + self.cfg.TRAINER.DISTILL_W2*loss_distill
                
                return loss  

        return logits


def _get_clones(module, N, M):
    return nn.ModuleList(nn.ModuleList([copy.deepcopy(module) for i in range(N)]) for j in range(M))


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=1)
   
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights


@TRAINER_REGISTRY.register()
class DCG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DCG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        print("Loading original CLIP for distillation")
        design_details = {
            "trainer": "CoOp",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        clip_model_distill = load_clip_to_cpu(cfg, design_details=design_details)

        if cfg.TRAINER.DCG.PREC == "fp32" or cfg.TRAINER.DCG.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            clip_model_distill.float()

        with open(
            f"gpt_file/{dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json"
        ) as f:
            gpt3_prompt = json.load(f)

        # Textual features
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_distill, cfg.DATASET.NAME
        )
        
        
        
        print("Building custom CLIP")
        self.model = CustomCLIP(
            cfg, classnames, clip_model, clip_model_distill, clip_weights
        )
        
        #total_params = sum(p.numel() for p in self.model.parameters())
#         print(f"Total trainable parameters: {total_params}")

        print("Turning off gradients in both the image and the text encoder")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
        
        
        if cfg.DATASET.SUBSAMPLE_CLASSES=="base":
            name_to_update = ["prompt_learner", "adapter"]
            for name, param in self.model.named_parameters():
                for n2u in name_to_update:
                    if n2u in name:
                        param.requires_grad_(True)
        else:
            name_to_update = ["prompt_learner", "adapter"]
            for name, param in self.model.named_parameters():
                for n2u in name_to_update:
                    if n2u in name:
                        param.requires_grad_(True)
        
        enabled = set()
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "MultiModalPromptLearner", self.model, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAINER.DCG.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image1, image2,image3,image4,image5, label = self.parse_batch_train(batch)
        
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.DCG.PREC
        if prec == "amp":
            with autocast():
                loss = model(image1, image2,image3,image4,image5,label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image1, image2,image3,image4,image5, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    def get_model(self):
        return self.model_inference
    

    def parse_batch_train(self, batch):
        input = batch["img"]
        image1, image2,image3,image4,image5 = input[0], input[1],input[2],input[3],input[4]
        label = batch["label"]
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        image3 = image3.to(self.device)
        image4 = image4.to(self.device)
        image5 = image5.to(self.device)
        label = label.to(self.device)
        return image1, image2,image3,image4,image5,label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]


            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def load_pre_trained(self, model_path):
        if not osp.exists(model_path):
            raise FileNotFoundError('Model not found at "{}"'.format(model_path))

        checkpoint = torch.load(model_path)
        missing_keys = self.model.load_state_dict(checkpoint, strict=False)

        if len(missing_keys.missing_keys) > 0:
            print("Missing keys: {}".format(missing_keys.missing_keys))
        if len(missing_keys.unexpected_keys) > 0:
            print("Unexpected keys: {}".format(missing_keys.unexpected_keys))
            
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader  
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
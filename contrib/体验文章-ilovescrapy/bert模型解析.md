# bert模型解析
BERT这个模型与其它两个不同的是

它在训练双向语言模型时以减小的概率把少量的词替成了Mask或者另一个随机的词。我个人感觉这个目的在于使模型被迫增加对上下文的记忆。至于这个概率，我猜是Jacob拍脑袋随便设的。
增加了一个预测下一句的loss。这个看起来就比较新奇了。
BERT模型具有以下两个特点：

第一，是这个模型非常的深，12层，并不宽(wide），中间层只有1024，而之前的Transformer模型中间层有2048。这似乎又印证了计算机图像处理的一个观点——深而窄 比 浅而宽 的模型更好。

第二，MLM（Masked Language Model），同时利用左侧和右侧的词语，这个在ELMo上已经出现了，绝对不是原创。其次，对于Mask（遮挡）在语言模型上的应用，已经被Ziang Xie提出了，用这篇论文的方法去做Masking，相信BRET的能力说不定还会有提升。
二、如何理解BERT模型
[1] BERT 要解决什么问题？

通常情况 transformer 模型有很多参数需要训练。譬如 BERT BASE 模型: L=12, H=768, A=12, 需要训练的模型参数总数是 12 * 768 * 12 = 110M。这么多参数需要训练，自然需要海量的训练语料。如果全部用人力标注的办法，来制作训练数据，人力成本太大。

受《A Neural Probabilistic Language Model》论文的启发，BERT 也用 unsupervised 的办法，来训练 transformer 模型。神经概率语言模型这篇论文，主要讲了两件事儿，1. 能否用数值向量（word vector）来表达自然语言词汇的语义？2. 如何给每个词汇，找到恰当的数值向量？
这篇论文写得非常精彩，深入浅出，要言不烦，而且面面俱到。经典论文，值得反复咀嚼。很多同行朋友都熟悉这篇论文，内容不重复说了。常用的中文汉字有 3500 个，这些字组合成词汇，中文词汇数量高达 50 万个。假如词向量的维度是 512，那么语言模型的参数数量，至少是 512 * 50万 = 256M

模型参数数量这么大，必然需要海量的训练语料。从哪里收集这些海量的训练语料？《A Neural Probabilistic Language Model》这篇论文说，每一篇文章，天生是训练语料。难道不需要人工标注吗？回答，不需要。
深度学习四大要素，1. 训练数据、2. 模型、3. 算力、4. 应用。训练数据有了，接下去的问题是模型。

 

[2] BERT 的五个关键词 Pre-training、Deep、Bidirectional、Transformer、Language Understanding 分别是什么意思？

《A Neural Probabilistic Language Model》这篇论文讲的 Language Model，严格讲是语言生成模型（Language Generative Model），预测语句中下一个将会出现的词汇。语言生成模型能不能直接移用到其它 NLP 问题上去？
语言生成模型，能不能很好地解决上述问题？进一步问，有没有 “通用的” 语言模型，能够理解语言的语义，适用于各种 NLP 问题？BERT 这篇论文的题目很直白，《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》，一眼看去，就能猜得到这篇文章会讲哪些内容。

这个题目有五个关键词，分别是 Pre-training、Deep、Bidirectional、Transformers、和 Language Understanding。其中 pre-training 的意思是，作者认为，确实存在通用的语言模型，先用文章预训练通用模型，然后再根据具体应用，用 supervised 训练数据，精加工（fine tuning）模型，使之适用于具体应用。为了区别于针对语言生成的 Language Model，作者给通用的语言模型，取了一个名字，叫语言表征模型 Language Representation Model。
能实现语言表征目标的模型，可能会有很多种，具体用哪一种呢？作者提议，用 Deep Bidirectional Transformers 模型。假如给一个句子 “能实现语言表征[mask]的模型”，遮盖住其中“目标”一词。从前往后预测[mask]，也就是用“能/实现/语言/表征”，来预测[mask]；或者，从后往前预测[mask]，也就是用“模型/的”，来预测[mask]，称之为单向预测 unidirectional。单向预测，不能完整地理解整个语句的语义。于是研究者们尝试双向预测。把从前往后，与从后往前的两个预测，拼接在一起 [mask1/mask2]，这就是双向预测 bi-directional。
模型架构

BERT的模型架构是基于Vaswani et al. (2017) 中描述的原始实现multi-layer bidirectional Transformer编码器，并在tensor2tensor库中发布。由于Transformer的使用最近变得无处不在，论文中的实现与原始实现完全相同，因此这里将省略对模型结构的详细描述。

在这项工作中，论文将层数（即Transformer blocks）表示为L，将隐藏大小表示为H，将self-attention heads的数量表示为A。在所有情况下，将feed-forward/filter 的大小设置为 4H，即H = 768时为3072，H = 1024时为4096。论文主要报告了两种模型大小的结果：

 : L=12, H=768, A=12, Total Parameters=110M
 : L=24, H=1024, A=16, Total Parameters=340M
为了进行比较，论文选择了  ，它与OpenAI GPT具有相同的模型大小。然而，重要的是，BERT Transformer 使用双向self-attention，而GPT Transformer 使用受限制的self-attention，其中每个token只能处理其左侧的上下文。研究团队注意到，在文献中，双向 Transformer 通常被称为“Transformer encoder”，而左侧上下文被称为“Transformer decoder”，因为它可以用于文本生成。BERT，OpenAI GPT和ELMo之间的比较如图1所示。

预训练模型架构的差异。BERT使用双向Transformer。OpenAI GPT使用从左到右的Transformer。ELMo使用经过独立训练的从左到右和从右到左LSTM的串联来生成下游任务的特征。三个模型中，只有BERT表示在所有层中共同依赖于左右上下文。

输入表示（input representation）

论文的输入表示（input representation）能够在一个token序列中明确地表示单个文本句子或一对文本句子（例如， [Question, Answer]）。对于给定token，其输入表示通过对相应的token、segment和position embeddings进行求和来构造。图2是输入表示的直观表示：

与Peters et al. (2018) 和 Radford et al. (2018)不同，论文不使用传统的从左到右或从右到左的语言模型来预训练BERT。相反，使用两个新的无监督预测任务对BERT进行预训练。

任务1: Masked LM

从直觉上看，研究团队有理由相信，深度双向模型比left-to-right 模型或left-to-right and right-to-left模型的浅层连接更强大。遗憾的是，标准条件语言模型只能从左到右或从右到左进行训练，因为双向条件作用将允许每个单词在多层上下文中间接地“see itself”。

为了训练一个深度双向表示（deep bidirectional representation），研究团队采用了一种简单的方法，即随机屏蔽（masking）部分输入token，然后只预测那些被屏蔽的token。论文将这个过程称为“masked LM”(MLM)，尽管在文献中它经常被称为Cloze任务(Taylor, 1953)。

在这个例子中，与masked token对应的最终隐藏向量被输入到词汇表上的输出softmax中，就像在标准LM中一样。在团队所有实验中，随机地屏蔽了每个序列中15%的WordPiece token。与去噪的自动编码器（Vincent et al.， 2008）相反，只预测masked words而不是重建整个输入。

虽然这确实能让团队获得双向预训练模型，但这种方法有两个缺点。首先，预训练和finetuning之间不匹配，因为在finetuning期间从未看到[MASK]token。为了解决这个问题，团队并不总是用实际的[MASK]token替换被“masked”的词汇。相反，训练数据生成器随机选择15％的token。例如在这个句子“my dog is hairy”中，它选择的token是“hairy”。然后，执行以下过程：

数据生成器将执行以下操作，而不是始终用[MASK]替换所选单词：

80％的时间：用[MASK]标记替换单词，例如，my dog is hairy → my dog is [MASK]
10％的时间：用一个随机的单词替换该单词，例如，my dog is hairy → my dog is apple
10％的时间：保持单词不变，例如，my dog is hairy → my dog is hairy. 这样做的目的是将表示偏向于实际观察到的单词。
Transformer encoder不知道它将被要求预测哪些单词或哪些单词已被随机单词替换，因此它被迫保持每个输入token的分布式上下文表示。此外，因为随机替换只发生在所有token的1.5％（即15％的10％），这似乎不会损害模型的语言理解能力。

使用MLM的第二个缺点是每个batch只预测了15％的token，这表明模型可能需要更多的预训练步骤才能收敛。团队证明MLM的收敛速度略慢于 left-to-right的模型（预测每个token），但MLM模型在实验上获得的提升远远超过增加的训练成本。

 

任务2：下一句预测

许多重要的下游任务，如问答（QA）和自然语言推理（NLI）都是基于理解两个句子之间的关系，这并没有通过语言建模直接获得。

在为了训练一个理解句子的模型关系，预先训练一个二进制化的下一句测任务，这一任务可以从任何单语语料库中生成。具体地说，当选择句子A和B作为预训练样本时，B有50％的可能是A的下一个句子，也有50％的可能是来自语料库的随机句子。例如：

Input = [CLS] the man went to [MASK] store [SEP]

he bought a gallon [MASK] milk [SEP]

Label = IsNext

Input = [CLS] the man [MASK] to the store [SEP]

penguin [MASK] are flight ##less birds [SEP]

Label = NotNext

团队完全随机地选择了NotNext语句，最终的预训练模型在此任务上实现了97％-98％的准确率。

BERT是一个语言表征模型（language representation model），通过超大数据、巨大模型、和极大的计算开销训练而成，在11个自然语言处理的任务中取得了最优（state-of-the-art, SOTA）结果。或许你已经猜到了此模型出自何方，没错，它产自谷歌。估计不少人会调侃这种规模的实验已经基本让一般的实验室和研究员望尘莫及了，但它确实给我们提供了很多宝贵的经验：

深度学习就是表征学习 （Deep learning is representation learning）："We show that pre-trained representations eliminate the needs of many heavily engineered task-specific architectures". 在11项BERT刷出新境界的任务中，大多只在预训练表征（pre-trained representation）微调（fine-tuning）的基础上加一个线性层作为输出（linear output layer）。在序列标注的任务里（e.g. NER），甚至连序列输出的依赖关系都先不管（i.e. non-autoregressive and no CRF），照样秒杀之前的SOTA，可见其表征学习能力之强大。
规模很重要（Scale matters）："One of our core claims is that the deep bidirectionality of BERT, which is enabled by masked LM pre-training, is the single most important improvement of BERT compared to previous work". 这种遮挡（mask）在语言模型上的应用对很多人来说已经不新鲜了，但确是BERT的作者在如此超大规模的数据+模型+算力的基础上验证了其强大的表征学习能力。这样的模型，甚至可以延伸到很多其他的模型，可能之前都被不同的实验室提出和试验过，只是由于规模的局限没能充分挖掘这些模型的潜力，而遗憾地让它们被淹没在了滚滚的paper洪流之中。
预训练价值很大（Pre-training is important）："We believe that this is the first work to demonstrate that scaling to extreme model sizes also leads to large improvements on very small-scale tasks, provided that the model has been sufficiently pre-trained". 预训练已经被广泛应用在各个领域了（e.g. ImageNet for CV, Word2Vec in NLP），多是通过大模型大数据，这样的大模型给小规模任务能带来的提升有几何，作者也给出了自己的答案。BERT模型的预训练是用Transformer做的，但我想换做LSTM或者GRU的话应该不会有太大性能上的差别，当然训练计算时的并行能力就另当别论了。

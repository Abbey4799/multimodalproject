# multimodalproject

大规模知识图谱已经成为问答、生成等自然语言处理任务（Natural Language Processing, NLP）的重要支持。
然而，大部分知识图谱的构建工作仅仅侧重于数据中的文本信息，忽视了其他模态提供的丰富信息。
因此，本小组构建了多模态知识图谱CN-Imagepedia，基于百科图文对，为大规模中文知识图谱CN-DBpedia的实体补全细粒度的图片信息。通过将知识图谱中的实体与多模态数据关联，多模态知识图谱有多种应用场景。本小组利用SentenceBERT自监督地构建了基于主题的多模态实体链接AMEL (Aspect-based Multimodal Entity Linking)数据集，以情境化的实体链接任务为例，展示了多模态知识图谱的应用潜力。

代码结构如下：

```
|-- multimodalproject
    |-- crawler		#爬取数据
		|-- method		#AMEL (Aspect-based Multimodal Entity Linking)数据集构建
    |-- dataset		#本项目构建的数据集
```


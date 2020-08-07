# TextRank
python 3 TextRank implemenation (english text data/ using pretrained word vector)

* requirements
  * spacy 
  * networkx 

* Using spacy pretrained word vector
 1) small word vector
  ```
  python -m spacy download en_core_web_sm 
  ```
  
 2) large word vector
  ```
  python -m spacy download en_core_web_lg 
  ```
## Usage
#### 1) Keyword extraction
```
python main.py --mode==keyword --keyword_top_k==10 
```
#### 2) Sentence extraction
```
python main.py --mode==sentence --sentece_top_k==5 
```

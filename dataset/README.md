# Conjunto de dados sobre posicionamentos

## Twitter

Comentários de usuários sobre operações policiais de grande repercussão no Brasil entre 2020 e 2021 veiculados simultaneamente nos portais de notícias G1, UOL e Folha de São Paulo obtidos da rede social Twitter (atual X) e anotados com posicionamentos de aprovação, desaprovação e neutralidade dos usuários sobre as operações

### Coleta de Dados

A coleta de dados foi realizada em duas etapas. Inicialmente, selecionamos notícias de grande repercussão sobre incidentes de segurança com intervenção policial entre 04/2019 e 12/2021 em três portais de notícias nacionais: UOL, Folha de São Paulo, e G1. As notícias selecionadas tinham mais de 700 comentários, indicando uma grande repercussão nas redes sociais.

Na segunda etapa, coletamos comentários do Twitter sobre as notícias selecionadas usando os títulos e links de cada notícia. Utilizamos a API do Twitter versão 2 com a biblioteca Tweepy em Python para coletar os tweets. Ao final dessa etapa, coletamos um total de 16.276 tweets.

### Pré-processamento e Rotulação

Realizamos um mínimo de pré-processamento nos tweets coletados, removendo links e quebras de linha e desconsiderando tweets com apenas uma palavra. Mantivemos todas as stop words para treinar modelos de inferência com a mesma sequência em que as palavras aparecem nos tweets.

Após o pré-processamento, realizamos a rotulação das instâncias de dados. O rótulo consiste na classificação de um tweet como Aprova, Desaprova ou Neutro, considerando o posicionamento do usuário que postou sobre a ação policial no incidente ao qual a notícia se refere. Contamos com três pessoas para rotular um subconjunto de 4.467 tweets definidos aleatoriamente e proporcionalmente ao total de tweets por notícia.

### Rotulação manual (dataset_atualizado_08022022.csv)

#### Descrição

Este dataset contém 4.467 tweets, selecionados aleatoriamente e proporcionalmente ao total de tweets por notícia. Os tweets foram rotulados por três pessoas não especialistas, considerando o posicionamento do usuário que postou sobre a ação policial no incidente ao qual a notícia se refere. Os rótulos consistem em "Aprova", "Desaprova" ou "Neutro". Este dataset serve de base para o treinamento de modelos de inferência.

#### Atributos

O dataset é separado por vírgula (`,`) e contém os seguintes atributos:

1. **id**: Identificador único para cada comentário no dataset.
2. **ano**: A data em que o comentário foi feito.
3. **site**: O site de onde o comentário foi coletado.
4. **comentarios**: O texto do comentário coletado.
5. **rotulacao_manual**: A rotulação manual do comentário. Um valor de 0 indica um comentário neutro,
6. 1 indica aprovação, enquanto -1 indica desaprovação em relação à operação polícial em questão.
7. **id_rotulador**: O identificador do rotulador que rotulou o comentário.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).


## YouTube

### Coleta de Dados

A partir da API do YouTube versão 3 desenvolvemos um programa coletor de comentários no referido contexto de forma automatizada. Esse coletor tem dois procedimentos que são (1) busca por eventos e (2) coleta de comentários sobre esses eventos. No primeiro procedimento utilizamos as seguintes palavras-chave: assassinato ou morte ou roubo ou furto ou polícia em períodos semanais. Assim, obtivemos uma série de 52 unidades de tempo (semanas) em 2022. No segundo procedimento codificamos no coletor, passamos o identificador do vídeo como parâmetro para Coletar informações de todos os comentários disponíveis para cada vídeo.

### Pré-processamento

Aplicamos um novo filtro sobre os vídeos coletados para selecionar apenas aqueles que têm relação com operações policial, que é o foco do nosso trabalho. Para isso, selecionamos apenas vídeos que continham no título as palavras “policial”, “pm” e “policia”, considerando transformação de todos os caracteres dos títulos em minúsculo e remoção de acentos e sinais ortográficos. A seguir, conduzimos outra série de pré-processamentos nos comentários via a biblioteca NLTK, removendo links, emojis e quebras de linha, mas mantendo stopwords para inferências com a mesma sequência em que as palavras aparecem nos comentários.

<!-- 2021 -->
### Comentários do ano de 2021 (comentarios_2021.csv):

#### Descrição

Este dataset contém **116.757 comentários** coletados de vídeos durante o ano de **2021**.

#### Atributos

O dataset é separado por tabulação (`\t`) e contém os seguintes atributos:

1. **id_comentario**: Identificador único para cada comentário no dataset.
2. **comentario**: O texto do comentário coletado.
3. **timestamp**: A data e hora em que o comentário foi postado, no formato ISO 8601.
4. **canal**: O canal do YouTube de onde o comentário foi coletado.
5. **curtidas**: O número de curtidas que o comentário recebeu.
6. **id_video**: O identificador do vídeo do qual o comentário foi coletado.
7. **id_comentario_pai**: O identificador do comentário pai, se o comentário for uma resposta a outro comentário.
8. **data_postagem**: A data e hora em que o comentário foi postado, no formato local.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).

### Comentários do ano de 2021 no Nordeste do Brasil (comentarios_2021_nordeste.csv):

#### Descrição

Este dataset contém **2.748 comentários** coletados de vídeos publicados na região do **Nordeste Brasileiro** durante o ano de **2021**. A coleta dos comentários foi realizada utilizando a mesma metodologia geral que os outros conjuntos de dados, com a única alteração sendo o local da coleta.

#### Atributos

O dataset é separado por tabulação (`\t`) e contém os seguintes atributos:

1. **id_comentario**: Identificador único para cada comentário no dataset.
2. **comentario**: O texto do comentário coletado.
3. **timestamp**: A data e hora em que o comentário foi postado, no formato ISO 8601.
4. **canal**: O canal do YouTube de onde o comentário foi coletado.
5. **curtidas**: O número de curtidas que o comentário recebeu.
6. **id_video**: O identificador do vídeo do qual o comentário foi coletado.
7. **id_comentario_pai**: O identificador do comentário pai, se o comentário for uma resposta a outro comentário.
8. **data_postagem**: A data e hora em que o comentário foi postado, no formato local.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).


### Comentários do ano de 2021 no Sudeste do Brasil (comentarios_2021_sudeste.csv):

#### Descrição

Este dataset contém **6.854 comentários** coletados de vídeos publicados na região do **Sudeste Brasileiro** durante o ano de **2021**. A coleta dos comentários foi realizada utilizando a mesma metodologia geral que os outros conjuntos de dados, com a única alteração sendo o local da coleta.

#### Atributos

O dataset é separado por tabulação (`\t`) e contém os seguintes atributos:

1. **id_comentario**: Identificador único para cada comentário no dataset.
2. **comentario**: O texto do comentário coletado.
3. **timestamp**: A data e hora em que o comentário foi postado, no formato ISO 8601.
4. **canal**: O canal do YouTube de onde o comentário foi coletado.
5. **curtidas**: O número de curtidas que o comentário recebeu.
6. **id_video**: O identificador do vídeo do qual o comentário foi coletado.
7. **id_comentario_pai**: O identificador do comentário pai, se o comentário for uma resposta a outro comentário.
8. **data_postagem**: A data e hora em que o comentário foi postado, no formato local.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).

<!-- 2022 -->
### Comentários do ano de 2022 no Nordeste do Brasil (comentarios_2022_nordeste.csv):

#### Descrição

Este dataset contém **8.552 comentários** coletados de vídeos publicados na região do **Nordeste Brasileiro** durante o ano de **2022**. A coleta dos comentários foi realizada utilizando a mesma metodologia geral que os outros conjuntos de dados, com a única alteração sendo o local da coleta.

#### Atributos

O dataset é separado por tabulação (`\t`) e contém os seguintes atributos:

1. **id_comentario**: Identificador único para cada comentário no dataset.
2. **comentario**: O texto do comentário coletado.
3. **timestamp**: A data e hora em que o comentário foi postado, no formato ISO 8601.
4. **canal**: O canal do YouTube de onde o comentário foi coletado.
5. **curtidas**: O número de curtidas que o comentário recebeu.
6. **id_video**: O identificador do vídeo do qual o comentário foi coletado.
7. **id_comentario_pai**: O identificador do comentário pai, se o comentário for uma resposta a outro comentário.
8. **data_postagem**: A data e hora em que o comentário foi postado, no formato local.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).

### Comentários do ano de 2022 no Sudeste do Brasil (comentarios_2022_sudeste.csv):

#### Descrição

Este dataset contém **24.643 comentários** coletados de vídeos publicados na região do **Sudeste Brasileiro** durante o ano de **2022**. A coleta dos comentários foi realizada utilizando a mesma metodologia geral que os outros conjuntos de dados, com a única alteração sendo o local da coleta.

#### Atributos

O dataset é separado por tabulação (`\t`) e contém os seguintes atributos:

1. **id_comentario**: Identificador único para cada comentário no dataset.
2. **comentario**: O texto do comentário coletado.
3. **timestamp**: A data e hora em que o comentário foi postado, no formato ISO 8601.
4. **canal**: O canal do YouTube de onde o comentário foi coletado.
5. **curtidas**: O número de curtidas que o comentário recebeu.
6. **id_video**: O identificador do vídeo do qual o comentário foi coletado.
7. **id_comentario_pai**: O identificador do comentário pai, se o comentário for uma resposta a outro comentário.
8. **data_postagem**: A data e hora em que o comentário foi postado, no formato local.

#### Uso

Este dataset é destinado a ser usado no repositório. Para mais informações sobre como usar este dataset no contexto desse projeto, consulte a documentação no [crimes_stance](https://github.com/LABPAAD/crimes_stance.git).
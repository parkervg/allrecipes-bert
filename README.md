BERT with a masked language model head was trained on the first 10,000 recipes in the AllRecipes dataset. The embeddings of all the nouns in `EPIC_noun_classes.csv` were then clustered with KMeans and annotated with the most frequently accompanying verbs in `EPIC_train_action_labels.csv`.

## Usage
- For clustering: `python3 lib/cluster_embeds.py {n_clusters} {noun_cap}`
  - Example: `python3 lib/cluster_embeds.py 15 50`
- For masked language modeling: `python3 lib/mlm.py {text} {results_length}`
  - Example: `python3 lib/mlm.py "Next, we want to [MASK] the salmon into thin fillets." 3`

## Misc. Notes to Self
  - Use masked language model to disambiguate verbs not in our vocab
    - Have probability threshold required to accept it
  - Use either clusters or distance from verbs as prior assumptions about the actions allowed to certain nouns
  - Use knowledge of a known object being able to be used in a certain way to guide interpretation of foreign object

## Cluster Examples:
  - [Cluster 14](#cluster-14): Ingredients that you can 'cut'
  - [Cluster 6](#cluster-6): Small utensils you can 'wash', 'take'
  - [Cluster 10](#cluster-10) and [Cluster 11](#cluster-11): Spices you can 'sprinkle', 'put' (on)
  - [Cluster 13](#cluster-13): Appliances you can 'open' and 'close'
  - Issues:
    - Examples like [Cluster 3](#cluster-3): semantic genre outside of affordances
    - Clusters like Cluster 0 are too broad, not that meaningful
    - Maybe use SHAP with classification to extract subspace related to allowed actions



coherance_score = sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)

Kmeans with k = 15

# Clustering Results
Embeddings were clustered with k = 15.

The 'Cluster coherence score' is defined as:

`sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)`.

'Top Verbs' list the 5 most frequent verbs attached to the clustered nouns in EpicKitchens and their intra-cluster frequencies.


## CLUSTER 0
#### Cluster coherence score: 0.667
#### Top Verbs: [('put', 28), ('take', 26), ('wash', 11), ('cut', 7), ('pour', 6)]
- butter
- light
- oil
- handle
- salad
- finger
- bowl
- chair
- paper
- watch
- rice
- fork
- bacon
- power
- towel
- heat
- spoon
- coconut
- brush
- cover
- chicken
- time
- banana
- hand
- cake
- ginger
- fire
- cd
- beer
- apple
- salt
- fruit
- knife
- microwave
- drink
- garlic
- basil
- cherry
- dust
- coffee
- water


## CLUSTER 1
#### Cluster coherence score: 1.0
#### Top Verbs: [('put', 4), ('take', 4), ('remove', 1), ('wash', 1), ('fold', 1)]
- plastic wrap
- ladle
- tablecloth
- whisk


## CLUSTER 2
#### Cluster coherence score: 0.778
#### Top Verbs: [('take', 8), ('put', 6), ('cut', 4), ('scoop', 2), ('dry', 1)]
- waffle
- caper
- salami
- risotto
- omelette
- bouquet garni
- kiwi
- aubergine
- courgette
- pancake


## CLUSTER 3
#### Cluster coherence score: 0.917
#### Top Verbs: [('take', 3), ('open', 3), ('put', 3), ('squeeze', 1), ('close', 1)]
- casserole
- gravy
- ketchup
- mayonnaise
- pepper shaker


## CLUSTER 4
#### Cluster coherence score: 0.797
#### Top Verbs: [('take', 18), ('put', 17), ('wash', 11), ('turn-on', 3), ('open', 2)]
- chopping board
- grinder
- pestle
- extractor fan
- blender
- utensil
- lime squeezer
- toaster
- remote control
- trouser
- spatula
- potato peeler
- cutlery
- rolling pin
- tongs
- presser
- grater
- masher
- sticker
- lime juicer
- slicer
- food processor
- strainer
- spot remover


## CLUSTER 5
#### Cluster coherence score: 0.664
#### Top Verbs: [('take', 57), ('put', 54), ('wash', 14), ('cut', 10), ('open', 9)]
- envelope
- roll
- peach
- fridge
- cream
- chip
- cup
- rest
- can
- sugar
- phone
- plate
- top
- cloth
- base
- curry
- table
- sink
- milk
- egg
- button
- napkin
- tomato
- book
- bread
- kitchen
- label
- tea
- jar
- heart
- skin
- stock
- grill
- wall
- pizza
- carrot
- lead
- pepper
- mint
- tray
- air
- flour
- fish
- crisp
- seed
- pot
- alarm
- wine
- cereal
- flame
- corn
- bag
- part
- window
- meat
- soap
- pie
- juice
- content
- tablet
- lemon
- cheese
- wrap
- berry
- tap
- chocolate
- tuna
- switch
- soup
- turkey
- glass
- lime
- apron
- potato
- pasta
- form
- onion
- honey
- support


## CLUSTER 6
#### Cluster coherence score: 0.781
#### Top Verbs: [('put', 9), ('take', 9), ('wash', 5), ('remove', 1), ('pour', 1)]
- oven glove
- colander
- coffee maker
- filter holder
- dust pan
- washing powder
- drying rack
- washing liquid
- rinse
- kitchen towel
- soap dish
- washing machine


## CLUSTER 7
#### Cluster coherence score: 0.682
#### Top Verbs: [('take', 54), ('put', 52), ('wash', 16), ('open', 14), ('close', 10)]
- wire
- funnel
- pan
- liquid
- lid
- vegetable
- leaf
- plug
- filter
- cabbage
- rosemary
- tube
- sock
- grape
- oven
- sponge
- dough
- squash
- cap
- spice
- vinegar
- mesh
- candle
- lamp
- rubbish
- container
- olive
- rim
- desk
- bottle
- cork
- pear
- scrap
- cinnamon
- scissors
- sandwich
- clip
- syrup
- bin
- rubber
- sleeve
- jug
- yeast
- mat
- knob
- drawer
- beef
- floor
- ladder
- poster
- herb
- rug
- boxer
- pea
- tail
- sausage
- straw
- package
- mustard
- instruction
- clothes
- almond
- foil
- salmon
- lighter
- shelf
- timer
- sauce
- kettle
- scale
- mixture
- sheets
- mushroom
- recipe
- shirt
- basket
- tissue
- coke
- ingredient
- box
- cupboard
- food


## CLUSTER 8
#### Cluster coherence score: 0.741
#### Top Verbs: [('put', 8), ('take', 7), ('cut', 2), ('open', 2), ('water', 1)]
- sushi mat
- spring onion
- onion ring
- salad dressing
- oat
- spinach
- parsley
- tuna burger
- garlic paste


## CLUSTER 9
#### Cluster coherence score: 0.688
#### Top Verbs: [('put', 24), ('take', 22), ('cut', 8), ('pour', 5), ('wash', 5)]
- leftover
- screw driver
- cereal bar
- noodle
- thyme
- whetstone
- blueberry
- lemon grass
- pith
- green bean
- hummus
- hob
- nesquik
- pineapple
- pesto
- crab stick
- egg shell
- chopstick
- raisin
- smoothie
- melon
- biscuit
- yoghurt
- tofu
- paella
- sprout
- mint leaf
- nutella
- bottle opener
- leek
- cumin
- slipper
- pine nut
- dumpling


## CLUSTER 10
#### Cluster coherence score: 0.889
#### Top Verbs: [('put', 3), ('take', 2), ('pour', 1), ('close', 1), ('cut', 1)]
- chilli flake
- chilli
- paprika


## CLUSTER 11
#### Cluster coherence score: 0.857
#### Top Verbs: [('take', 5), ('put', 4), ('sprinkle', 1), ('open', 1), ('close', 1)]
- oregano
- coconut powder
- turmeric
- coriander
- cayenne pepper


## CLUSTER 12
#### Cluster coherence score: 1.0
#### Top Verbs: [('put', 2), ('take', 2), ('flip', 1), ('open', 1), ('serve', 1)]
- tortilla
- mocha
- jambalaya


## CLUSTER 13
#### Cluster coherence score: 0.833
#### Top Verbs: [('put', 3), ('insert', 2), ('open', 2), ('close', 2), ('move', 1)]
- slow cooker
- heater
- dishwasher
- freezer


## CLUSTER 14
#### Cluster coherence score: 0.938
#### Top Verbs: [('cut', 5), ('take', 4), ('put', 4), ('measure', 1), ('pour', 1)]
- celery
- cucumber
- breadcrumb
- lettuce
- broccoli
- avocado

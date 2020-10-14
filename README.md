Notes:
  - Use masked language model to disambiguate verbs not in our vocab
    - Have probability threshold required to accept it
  - Use either clusters or distance from verbs as prior assumptions about the actions allowed to certain nouns
  - Use knowledge of a known object being able to be used in a certain way to guide interpretation of foreign object
    - Examples:
  - [Cluster 5](#cluster-5) of non-food kitchen tools: 'take', 'wash'
  - [Cluster 9](#cluster-2): many spices
  - Issues:
    - Examples like [Cluster 14](#cluster-14): semantic genre outside of affordances
    - Maybe use SHAP with classification to extract subspace related to allowed actions



coherance_score = sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)

Kmeans with k = 15

## CLUSTER 0
#### Cluster coherence score: 0.571
#### Top Verbs: [('pick-up', 6), ('put-down', 5), ('put', 4), ('take', 3), ('wash', 2)]
- peeler
- pestle
- drainer
- brusher
- washer
- seasoning
- detergent
- utensil
- strainer
- presser
- spatula
- blender
- sticker
- burner
- grinder


## CLUSTER 1
#### Cluster coherence score: 0.498
#### Top Verbs: [('put', 50), ('pick-up', 32), ('take', 25), ('put-down', 16), ('open', 15)]
- bag
- butter
- cloth
- serving
- tablet
- soy
- potato
- mug
- cheese
- tuna
- scoop
- rolled
- crisp
- boil
- cream
- spade
- window
- soup
- frozen
- curry
- soda
- pie
- button
- switch
- vinegar
- sliced
- remote
- empty
- lemon
- steel
- jar
- chip
- toast
- flame
- bay
- cleaning
- pepper
- carrot
- shopping
- napkin
- draining
- milk
- left
- kitchen
- blade
- handle
- baking
- sink
- sugar
- roll
- cookie
- popcorn
- fold
- steak
- onion
- trash
- fry
- berry
- candle
- basil
- grill
- flour
- clothes
- sandwich
- corn
- skin
- case
- sun
- shrimp
- sheet
- peach
- lime
- pasta
- big
- juice
- tong
- small
- chocolate
- pizza
- tea
- split
- holder
- packing
- eating
- rolling
- rack
- dough
- dish
- washing
- cutting
- bean
- alarm
- drying
- cereal
- whole
- wrapping
- meat
- egg
- peeling
- peel
- few
- beef
- pot
- apron
- pouring
- tray
- ham
- phone
- balance
- honey
- wrap
- tap
- measuring
- form
- rubbish
- side
- my
- loaf
- tomato
- envelope
- dressing
- stock
- pastry
- oven
- plate
- fridge
- cinnamon
- pick
- gas
- letter
- canned


## CLUSTER 2
#### Cluster coherence score: 0.722
#### Top Verbs: [('put', 4), ('pick-up', 3), ('chop', 2), ('take', 2), ('put-down', 2)]
- yoghurt
- coriander
- oregano
- masher
- cilantro
- turmeric
- balsamic


## CLUSTER 3
#### Cluster coherence score: 0.491
#### Top Verbs: [('put', 39), ('pick-up', 23), ('take', 21), ('open', 14), ('put-down', 14)]
- heat
- ginger
- black
- seed
- third
- lead
- top
- banana
- bacon
- new
- sweet
- hand
- cooking
- chop
- dust
- towel
- second
- table
- fork
- spoon
- drink
- apple
- oil
- hot
- mobile
- seal
- cd
- label
- instant
- other
- fruit
- press
- can
- fish
- bread
- wood
- bowl
- pack
- support
- chinese
- mix
- book
- chicken
- spring
- wash
- film
- cake
- baby
- open
- glass
- chopped
- spray
- microwave
- work
- coffee
- light
- water
- chair
- close
- slow
- mint
- door
- round
- national
- air
- pitcher
- more
- salt
- soap
- red
- moon
- measure
- cut
- bell
- screw
- salad
- pour
- lunch
- salsa
- paper
- knife
- power
- wine
- right
- burger
- ground
- fire
- base
- time
- garlic
- watch
- beer
- turkey
- cabinet
- finger
- green
- pit
- part
- draw
- rice
- cherry
- brush
- refrigerator
- content
- only
- cover
- coconut
- take


## CLUSTER 4
#### Cluster coherence score: 0.667
#### Top Verbs: [('put', 2), ('mix-on', 1), ('mix', 1), ('sprinkle', 1), ('pick-up', 1)]
- sprout
- sweetcorn
- parsley


## CLUSTER 5
#### Cluster coherence score: 0.825
#### Top Verbs: [('pick-up', 8), ('put-down', 8), ('take', 8), ('put', 7), ('wash', 2)]
- silverware
- teapot
- degreaser
- saucer
- saucepan
- dustpan
- tupperware
- rinse
- diced
- tableware
- tongs
- tablecloth
- colander
- plated
- trouser
- ladle
- cutlery


## CLUSTER 6
#### Cluster coherence score: 0.55
#### Top Verbs: [('pick-up', 6), ('put-down', 2), ('take', 1), ('wash', 1), ('rinse', 1)]
- grater
- carafe
- mitt
- mitten
- temp
- chilli
- sieve
- fraiche
- creme


## CLUSTER 7
#### Cluster coherence score: 0.614
#### Top Verbs: [('put', 10), ('pick-up', 8), ('take', 3), ('open', 3), ('pour', 3)]
- smoothie
- chipping
- dumpling
- corncob
- raisin
- kiwi
- pancake
- cracker
- biscuit
- paprika
- waffle
- nesquik
- frying
- eggshell
- granola
- chopping
- teabag
- breadcrumb
- chopstick
- whisk


## CLUSTER 8
#### Cluster coherence score: 0.692
#### Top Verbs: [('put', 3), ('slice', 2), ('cut', 2), ('open', 1), ('close', 1)]
- cucumber
- mayonnaise
- broccoli
- celery
- casserole


## CLUSTER 9
#### Cluster coherence score: 0.545
#### Top Verbs: [('pick-up', 10), ('put', 7), ('take', 6), ('cut', 4), ('open', 3)]
- hummus
- rind
- mirin
- paneer
- nutella
- cling
- pesto
- extractor
- saag
- rusk
- melba
- blueberry
- cumin
- galangal
- hob
- leek
- cob
- pineapple
- clove
- wok
- pith
- sachet
- jeera
- melon


## CLUSTER 10
#### Cluster coherence score: 0.667
#### Top Verbs: [('pick-up', 4), ('take', 2), ('put-down', 2), ('cut', 1), ('place', 1)]
- caper
- minced
- mincing
- spinach
- scissor
- shreddies
- mince
- thyme


## CLUSTER 11
#### Cluster coherence score: 0.51
#### Top Verbs: [('pick-up', 6), ('take', 6), ('put', 5), ('throw', 4), ('put-down', 4)]
- wrapper
- dishrag
- passata
- dustbin
- lasagne
- stopper
- tofu
- leftover
- trivet
- nozzle
- flatware
- dishing
- wastebasket
- paella
- noodle
- cleanser
- stirrer
- scrubber
- dishwater
- sharpener
- v60
- slipper
- spinker


## CLUSTER 12
#### Cluster coherence score: 0.8
#### Top Verbs: [('open', 2), ('put-down', 2), ('put', 2), ('close', 1), ('take', 1)]
- miso
- sushi
- moka
- masala
- jambalaya
- mocha
- soya
- tamagoyaki


## CLUSTER 13
#### Cluster coherence score: 1.0
#### Top Verbs: [('wash', 1), ('pick-up', 1), ('put-down', 1), ('rinse', 1)]
- tablespoon
- teaspoon


## CLUSTER 14
#### Cluster coherence score: 0.733
#### Top Verbs: [('put-down', 4), ('pick-up', 2), ('put', 2), ('take', 2), ('clean', 1)]
- mezzaluna
- prosciutto
- salami
- mozzarella
- spate
- ciabatta
- chorizo
- risotto

Notes:
  - Use masked language model to disambiguate verbs not in our vocab
    - Have probability threshold required to accept it
  - Use either clusters or distance from verbs as prior assumptions about the actions allowed to certain nouns
    - Examples:
      - spices clustered together, non-food items clustered together
      - Cluster 2 of appliances: 'take', 'put', etc. but no 'toss', 'pour', 'dice', etc.

    - Use knowledge of a known object being able to be used in a certain way to guide interpretation of foreign object
    - Maybe use SHAP with classification to extract subspace related to allowed actions


coherance_score = sum(counts for 5 most common verbs) / len(all verb tokens in the cluster)

## CLUSTER 0
#### Cluster coherence score: 0.484
#### Top Verbs: [('put', 52), ('take', 34), ('open', 26), ('pick-up', 24), ('close', 19)]
   box
   poster
   transparent
   holder
   crab
   tail
   canned
   sponge
   hop
   mat
   food
   filtered
   case
   rubber
   maker
   grocery
   ingredient
   socket
   pan
   remaining
   waste
   cleaner
   mushroom
   plug
   ladder
   sausage
   wire
   cap
   shirt
   lid
   filter
   mustard
   plain
   pine
   closet
   peeled
   rag
   bouquet
   own
   mixture
   recipe
   vinegar
   spice
   clothes
   cinnamon
   shelf
   rosemary
   clip
   herb
   convenient
   liquid
   cupboard
   instruction
   get
   paste
   maple
   bucket
   transferring
   olive
   tube
   flap
   kettle
   dough
   stove
   sesame
   container
   floor
   pea
   nut
   funnel
   tile
   laundry
   bottle
   basket
   mesh
   fan
   electronic
   fried
   pear
   broom
   dirty
   desk
   hamburger
   wall
   scale
   tin
   straw
   boxer
   vegetable
   flower
   drawer
   lamp
   rug
   spaghetti
   coke
   bin
   grape
   glove
   leaf
   salmon
   oven
   packet
   rim
   counter
   cabbage
   plastic
   flavour
   almond
   spot
   processor
   splitting
   outer
   cup
   machine
   foil
   board
   timer
   boiling
   lighter
   yeast
   patty
   sock
   surface
   knob
   shrimp
   bulb
   scrap
   chopper
   t
   beef
   white
   cork
   heart
   cardboard
   garbage
   excess
   weighing
   rest
   package
   temperature
   rocket
   extra
   locker
   tissue
   inner
   cooked
   pressure
   hood
   wooden
   sauce
   sleeve
   shell
   scissors
   packaging
   tidy
   jug
   squash


## CLUSTER 1
#### Cluster coherence score: 0.636
#### Top Verbs: [('put', 4), ('pick-up', 3), ('put-down', 3), ('cut', 2), ('take', 2)]
   avocado
   tupperware
   mozzarella
   turmeric
   prosciutto
   risotto
   tortilla
   lettuce
   mezzaluna


## CLUSTER 2
#### Cluster coherence score: 0.475
#### Top Verbs: [('pick-up', 7), ('put-down', 6), ('take', 6), ('put', 5), ('open', 4)]
   cooker
   presser
   toaster
   dishwasher
   blender
   washer
   peeler
   grinder
   masher
   ventilator
   drainer
   freezer
   burner
   strainer
   utensil
   sticker
   brusher
   grater
   spatula
   heater
   trouser
   cutlery
   pestle


## CLUSTER 3
#### Cluster coherence score: 0.509
#### Top Verbs: [('put', 51), ('pick-up', 32), ('take', 25), ('put-down', 15), ('open', 13)]
   serving
   tong
   sun
   egg
   tray
   window
   dish
   crisp
   small
   coconut
   split
   cleaning
   packing
   washing
   bag
   fold
   cream
   envelope
   form
   kitchen
   chip
   boil
   button
   trash
   cooking
   alarm
   wrap
   eating
   steel
   milk
   lime
   moon
   handle
   rack
   cutting
   scoop
   spade
   blade
   soap
   tap
   stock
   tea
   side
   berry
   big
   pick
   ham
   bay
   rolling
   flame
   napkin
   basil
   pizza
   peach
   cookie
   few
   jar
   toast
   tablet
   candle
   soy
   carrot
   cheese
   tomato
   shopping
   wrapping
   steak
   mug
   onion
   pepper
   phone
   potato
   apron
   grill
   pastry
   sandwich
   left
   chicken
   bean
   whole
   letter
   draining
   pouring
   balance
   cloth
   peel
   pie
   measuring
   pasta
   switch
   frozen
   flour
   sheet
   remote
   sugar
   roll
   sliced
   popcorn
   meat
   rolled
   lemon
   empty
   fridge
   sink
   my
   butter
   loaf
   fry
   pot
   curry
   peeling
   gas
   cereal
   drying
   honey
   corn
   chocolate
   juice
   skin
   plate
   garlic
   baking
   dressing
   soda
   tuna
   rubbish
   soup


## CLUSTER 4
#### Cluster coherence score: 0.635
#### Top Verbs: [('pick-up', 17), ('put', 14), ('take', 8), ('cut', 8), ('open', 7)]
   tamagoyaki
   miso
   pesto
   aubergine
   cafetiere
   mirin
   hummus
   leek
   moka
   granola
   chorizo
   thyme
   oregano
   melon
   ciabatta
   coriander
   salami
   kiwi
   nesquik
   cumin
   masala
   paprika
   jeera
   galangal
   spate
   sushi
   chilli
   pineapple
   pith
   courgette
   cayenne
   jambalaya
   soya
   mitten
   balsamic
   melba
   mocha
   omelette
   cilantro
   mitt


## CLUSTER 5
#### Cluster coherence score: 0.463
#### Top Verbs: [('pick-up', 12), ('put', 9), ('take', 8), ('put-down', 5), ('throw', 4)]
   spinker
   leftover
   creme
   rind
   sachet
   cob
   clove
   passata
   hob
   dishrag
   nutella
   saag
   stopper
   carafe
   temp
   dishwater
   blueberry
   flatware
   sharpener
   rusk
   noodle
   wok
   wastebasket
   sieve
   paella
   fraiche
   slipper
   tofu
   nozzle
   lasagne
   v60
   wrapper
   dustbin
   trivet
   paneer
   cling


## CLUSTER 6
#### Cluster coherence score: 0.667
#### Top Verbs: [('pick-up', 8), ('put-down', 8), ('put', 5), ('take', 4), ('spray', 3)]
   chipping
   dustpan
   ladle
   mincing
   scrubber
   frying
   rinse
   detergent
   diced
   chopstick
   seasoning
   extractor
   minced
   stirrer
   cleanser
   tableware
   shreddies
   plated
   chopping
   mince
   dumpling
   tongs
   dishing
   colander
   degreaser
   silverware


## CLUSTER 7
#### Cluster coherence score: 0.545
#### Top Verbs: [('put', 4), ('open', 2), ('cut', 2), ('pour', 2), ('slice', 2)]
   casserole
   mayonnaise
   broccoli
   cucumber
   ketchup
   celery
   oat
   gravy
   mincemeat


## CLUSTER 8
#### Cluster coherence score: 0.661
#### Top Verbs: [('put', 13), ('pick-up', 12), ('take', 9), ('put-down', 5), ('pour', 2)]
   whisk
   waffle
   caper
   raisin
   saucer
   pancake
   scissor
   yoghurt
   teapot
   tablespoon
   teaspoon
   eggshell
   sprout
   saucepan
   parsley
   sweetcorn
   cracker
   spinach
   teabag
   breadcrumb
   tablecloth
   corncob
   smoothie
   biscuit


## CLUSTER 9
#### Cluster coherence score: 0.491
#### Top Verbs: [('put', 35), ('pick-up', 23), ('take', 20), ('open', 14), ('put-down', 13)]
   measure
   work
   hand
   spring
   salt
   water
   time
   third
   spray
   banana
   sweet
   burger
   pour
   bell
   part
   heat
   wash
   film
   hot
   fish
   right
   chinese
   open
   lead
   close
   seed
   seal
   cabinet
   finger
   mobile
   more
   rice
   fruit
   chopped
   base
   label
   cherry
   microwave
   lunch
   bread
   cake
   content
   coffee
   baby
   paper
   mint
   watch
   black
   take
   chop
   slow
   salsa
   knife
   beer
   fork
   fire
   national
   drink
   bacon
   book
   red
   dust
   towel
   oil
   wine
   spoon
   wood
   other
   glass
   cover
   ginger
   instant
   light
   pitcher
   door
   chair
   bowl
   pack
   table
   ground
   mix
   salad
   new
   cd
   pit
   support
   second
   green
   turkey
   air
   power
   press
   top
   only
   refrigerator
   round
   screw
   brush
   draw
   apple
   cut
   can

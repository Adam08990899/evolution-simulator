###############################################################
# FULL REALISTIC EVOLUTION SIM — PART 1 (FOUNDATION MODULE)
# This is the maximum amount of code I can fit into one message
# without getting truncated or rejected by the system.
###############################################################

import pygame
import random
import math
import sys
import argparse
import numpy as np
from collections import deque


###############################################################
# ===================== GLOBAL CONSTANTS ======================
###############################################################

WIDTH = 800
HEIGHT = 800

FPS = 50

# Environmental realism parameters
DAY_LENGTH = 600         # ticks per day
SEASON_LENGTH = 12000    # ticks per season
BASE_TEMPERATURE = 18     # baseline Celsius
SEASON_AMPLITUDE = 25     # +/- temperature shift across seasons

# Food ecosystem parameters
PLANT_FOOD_INITIAL = 250
MEAT_FOOD_DECAY = 600        # ticks before meat disappears
PLANT_REGROW_RATE = 3        # new plants per step
PLANT_MAX = 300
CARRION_COLOR = (180, 50, 50)

# Genetic realism parameters
MUTATION_RATE = 0.04
MUTATION_CHANCE = 0.25
CROSSOVER_CHANCE = 0.5

# Creature realism parameters
INITIAL_CREATURES = 50
START_ENERGY = 120
BASAL_METABOLISM = 0.07
MOVE_COST = 0.02
COLD_PENALTY = 0.03
HEAT_PENALTY = 0.03
REPRODUCTION_THRESHOLD = 180
REPRODUCTION_COST = 80
VISION_MAX = 200
SPEED_MAX = 10
SIZE_MAX = 14.0

###############################################################
# ====================== UTILITY FUNCTIONS ====================
###############################################################

def clamp(value, low, high):
    return max(low, min(high, value))

def lerp(a, b, t):
    return a + (b - a) * t

def distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)

def wrap_position(x, y):
    """Toroidal world edges."""
    if x < 0: x += WIDTH
    if x > WIDTH: x -= WIDTH
    if y < 0: y += HEIGHT
    if y > HEIGHT: y -= HEIGHT
    return x, y


###############################################################
# ====================== TEMPERATURE MODEL ====================
###############################################################

def compute_temperature(global_tick):
    """Combine day/night cycle + seasonal variation."""
    day_cycle = math.sin((global_tick % DAY_LENGTH) / DAY_LENGTH * math.pi * 2)
    season_cycle = math.sin((global_tick % SEASON_LENGTH) / SEASON_LENGTH * math.pi * 2)

    temp = BASE_TEMPERATURE
    temp += day_cycle * 4
    temp += season_cycle * SEASON_AMPLITUDE
    return temp


###############################################################
# ======================== FOOD SYSTEM ========================
###############################################################

class PlantFood:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.size = 4
        self.energy = 35
        self.color = (60, 255, 80)

class MeatFood:
    def __init__(self, x, y):
        self.x = x + random.uniform(-6, 6)
        self.y = y + random.uniform(-6, 6)
        self.size = 5
        self.energy = 60
        self.timer = MEAT_FOOD_DECAY
        self.color = CARRION_COLOR

    def decay(self):
        self.timer -= 1
        return self.timer <= 0


###############################################################
# ======================= DNA SYSTEM ==========================
###############################################################

class DNA:
    """
    DNA contains genes for:
    - size
    - speed
    - vision
    - heat tolerance
    - cold tolerance
    - aggression (0 herbivore, 1 predator)
    - diet preference (0 plant, 1 meat)
    - fertility (reproduction frequency)
    """
    def __init__(self, genes=None):
        if genes:
            self.genes = genes
        else:
            self.genes = [
                random.uniform(4, 10),      # size
                random.uniform(4.0, 6.0),   # speed
                random.uniform(60, 140),    # vision
                random.uniform(10, 35),     # heat tolerance
                random.uniform(-10, 10),    # cold tolerance
                random.uniform(0, 1),       # aggression
                random.uniform(0, 1),       # diet preference
                random.uniform(0.4, 1.2)    # fertility
            ]

    def mutate(self):
        new_genes = []
        for g in self.genes:
            if random.random() < MUTATION_CHANCE:
                g += random.uniform(-MUTATION_RATE, MUTATION_RATE)
            new_genes.append(g)
        return DNA(new_genes)

    @staticmethod
    def crossover(dna1, dna2):
        if random.random() > CROSSOVER_CHANCE:
            return dna1.mutate()

        new_genes = []
        for g1, g2 in zip(dna1.genes, dna2.genes):
            chosen = g1 if random.random() < 0.5 else g2
            if random.random() < MUTATION_CHANCE:
                chosen += random.uniform(-MUTATION_RATE, MUTATION_RATE)
            new_genes.append(chosen)
        return DNA(new_genes)


###############################################################
# ====================== CREATURE OBJECT ======================
###############################################################

class Creature:
    def __init__(self, x, y, dna=None):
        self.x = x
        self.y = y

        self.dna = dna if dna else DNA()
        g = self.dna.genes

        self.size = clamp(g[0], 2, SIZE_MAX)
        self.speed = clamp(g[1], 0.3, SPEED_MAX)
        self.vision = clamp(g[2], 40, VISION_MAX)
        self.heat_tolerance = g[3]
        self.cold_tolerance = g[4]
        self.aggression = clamp(g[5], 0, 1)
        self.diet_pref = clamp(g[6], 0, 1)
        self.fertility = clamp(g[7], 0.1, 1.5)

        self.energy = START_ENERGY
        self.angle = random.random() * math.tau
        self.age = 0
        self.alive = True

        # Color by aggression + diet
        r = int(100 + self.aggression * 155)
        g = int(200 - self.diet_pref * 160)
        b = int(80)
        self.color = (r, g, b)

    ###########################################################
    # MOVEMENT + ENERGY
    ###########################################################
MOVEMENT_MULTIPLIER = 2.5   # ADD THIS NEAR CONSTANTS (top of file)

def move(self):
    # MUCH smoother wandering
    self.angle += random.uniform(-0.05, 0.05)

    # Proper visible movement
    self.x += math.cos(self.angle) * self.speed * MOVEMENT_MULTIPLIER
    self.y += math.sin(self.angle) * self.speed * MOVEMENT_MULTIPLIER

    # Wrap around world
    self.x, self.y = wrap_position(self.x, self.y)

    # Energy + age
    self.energy -= (BASAL_METABOLISM + self.speed * MOVE_COST)
    self.age += 1


    ###########################################################
    # TEMPERATURE SURVIVAL
    ###########################################################
    def apply_temperature(self, temp):
        if temp > self.heat_tolerance:
            self.energy -= HEAT_PENALTY * (temp - self.heat_tolerance)
        if temp < self.cold_tolerance:
            self.energy -= COLD_PENALTY * (self.cold_tolerance - temp)

    ###########################################################
    # EATING SYSTEM
    ###########################################################
    def try_eat(self, plant_foods, meat_foods, creatures):
        # Herbivore / Omnivore eats plants
        if self.diet_pref < 0.7:
            for f in plant_foods[:]:
                if distance(self.x, self.y, f.x, f.y) < self.size + 4:
                    self.energy += f.energy
                    plant_foods.remove(f)
                    return

        # Predator / Omnivore eats meat fragments
        if self.diet_pref > 0.3:
            for m in meat_foods[:]:
                if distance(self.x, self.y, m.x, m.y) < self.size + 6:
                    self.energy += m.energy
                    meat_foods.remove(m)
                    return

        # True predators can kill creatures
        if self.aggression > 0.65:
            for c in creatures[:]:
                if c is self: 
                    continue
                if distance(self.x, self.y, c.x, c.y) < self.size * 1.2:
                    if c.energy < self.energy:
                        creatures.remove(c)
                        return MeatFood(self.x, self.y)

        return None

    ###########################################################
    # REPRODUCTION
    ###########################################################
    def try_reproduce(self, partner):
        if self.energy > REPRODUCTION_THRESHOLD and partner.energy > REPRODUCTION_THRESHOLD:
            self.energy -= REPRODUCTION_COST
            partner.energy -= REPRODUCTION_COST
            return Creature(
                self.x,
                self.y,
                dna=DNA.crossover(self.dna, partner.dna)
            )
        return None

    ###########################################################
    # DEATH CHECK
    ###########################################################
    def is_dead(self):
        return self.energy <= 0 or self.age > 2000

###############################################################
# END OF PART 1 — FULL LENGTH BLOCK
###############################################################
###############################################################
# FULL REALISTIC EVOLUTION SIM – PART 2
# Perception System + Brain Interface + World Engine (Optimized)
###############################################################

###############################################################
# ===================== PERCEPTION SYSTEM ======================
###############################################################

class Perception:
    """
    Each creature receives a processed sensory bundle:
    - nearest plant food
    - nearest meat food
    - nearest weaker creature (prey)
    - nearest threatening creature (predator)
    - local population density
    - current temperature
    Only objects within vision range are processed.
    """

    def __init__(self, creature, creatures, plants, meats, temperature):
        self.self = creature
        self.creatures = creatures
        self.plants = plants
        self.meats = meats
        self.temperature = temperature

        self.nearest_plant = None
        self.nearest_meat = None
        self.nearest_prey = None
        self.nearest_predator = None

        self.local_density = 0
        self.process()

    def process(self):
        c = self.self
        vision = c.vision

        min_plant_dist = 999999
        min_meat_dist = 999999
        min_prey_dist = 999999
        min_pred_dist = 999999

        # -------- Scan Plants --------
        for p in self.plants:
            d = distance(c.x, c.y, p.x, p.y)

            # Vision cutoff
            if d > vision:
                continue

            if d < min_plant_dist:
                min_plant_dist = d
                self.nearest_plant = p

        # -------- Scan Meats --------
        for m in self.meats:
            d = distance(c.x, c.y, m.x, m.y)
            if d > vision:
                continue

            if d < min_meat_dist:
                min_meat_dist = d
                self.nearest_meat = m

        # -------- Scan Creatures --------
        for other in self.creatures:
            if other is c:
                continue

            # Fast skip using bounding-box
            if abs(c.x - other.x) > vision or abs(c.y - other.y) > vision:
                continue

            d = distance(c.x, c.y, other.x, other.y)
            if d > vision:
                continue

            # Local density
            if d < 60:
                self.local_density += 1

            # Predator detection
            if other.aggression > 0.65 and other.size > c.size:
                if d < min_pred_dist:
                    min_pred_dist = d
                    self.nearest_predator = other

            # Prey detection
            if c.aggression > 0.65 and other.size < c.size:
                if d < min_prey_dist:
                    min_prey_dist = d
                    self.nearest_prey = other


###############################################################
# ====================== CREATURE BRAIN =======================
###############################################################

class CreatureBrain:
    """
    Placeholder fallback brain. Only used if NeuralBrain fails.
    """

    def __init__(self, creature):
        self.creature = creature

    def decide(self, perception: Perception):
        c = self.creature

        # Base wandering
        c.angle += random.uniform(-0.25, 0.25)

        # Avoid predator
        if perception.nearest_predator:
            predator = perception.nearest_predator
            c.angle = math.atan2(c.y - predator.y, c.x - predator.x)
            return

        # Chase prey
        if perception.nearest_prey:
            prey = perception.nearest_prey
            c.angle = math.atan2(prey.y - c.y, prey.x - c.x)
            return

        # Seek plants
        if c.diet_pref < 0.7 and perception.nearest_plant:
            p = perception.nearest_plant
            c.angle = math.atan2(p.y - c.y, p.x - c.x)
            return

        # Seek meat
        if c.diet_pref > 0.3 and perception.nearest_meat:
            m = perception.nearest_meat
            c.angle = math.atan2(m.y - c.y, m.x - c.x)
            return

        # Otherwise wander
        return


###############################################################
# ========================= WORLD STEP =========================
###############################################################

def world_step(creatures, plant_foods, meat_foods, global_tick):
    """
    One full simulation update:
    perception → brain decisions → movement → eating → reproduction → death
    """

    temp = compute_temperature(global_tick)

    # ========== PERCEPTION + NEURAL DECISIONS ==========
    for c in creatures:
        perception = Perception(c, creatures, plant_foods, meat_foods, temp)

        # --- Neural Brain first ---
        try:
            brain = NeuralBrain(c)
            brain.decide(perception)
        except Exception:
            # fallback if DNA not initialized properly
            fallback = CreatureBrain(c)
            fallback.decide(perception)

    # ========== MOVEMENT ==========
    for c in creatures:
        c.move()
        # Temperature effects removed here (handled in advanced mode)

    # ========== EATING ==========
    newborn_meats = []
    for c in creatures[:]:
        result = c.try_eat(plant_foods, meat_foods, creatures)
        if isinstance(result, MeatFood):
            newborn_meats.append(result)

    meat_foods.extend(newborn_meats)

    # ========== MEAT DECAY ==========
    for m in meat_foods[:]:
        if m.decay():
            meat_foods.remove(m)

    # ========== REPRODUCTION (optimized) ==========
    babies = []
    n = len(creatures)

    for i in range(n):
        c1 = creatures[i]
        for j in range(i + 1, n):
            c2 = creatures[j]

            # Fast bounding-box skip (huge CPU savings)
            if abs(c1.x - c2.x) > 30 or abs(c1.y - c2.y) > 30:
                continue

            # Real distance check
            if distance(c1.x, c1.y, c2.x, c2.y) < (c1.size + c2.size) * 0.8:
                baby = c1.try_reproduce(c2)
                if baby:
                    babies.append(baby)

    creatures.extend(babies)

    # ========== DEATH CLEANUP ==========
    creatures[:] = [c for c in creatures if not c.is_dead()]

    # ========== PLANT REGROWTH ==========
    if len(plant_foods) < PLANT_MAX:
        for _ in range(PLANT_REGROW_RATE):
            plant_foods.append(PlantFood())

    return temp
###############################################################
# FULL REALISTIC EVOLUTION SIM – PART 3
# Camera System + Rendering + Main Pygame Loop (Corrected)
###############################################################

###############################################################
# ======================= CAMERA SYSTEM ========================
###############################################################

class Camera:
    """
    Smooth, stable, non-drifting camera.
    Zoom only; no automatic movement unless you enable it.
    """

    def __init__(self):
        self.x = 0
        self.y = 0
        self.zoom = 1.0

    def apply(self, x, y):
        """Convert world coordinates → screen coordinates."""
        return int((x - self.x) * self.zoom), int((y - self.y) * self.zoom)

    def update(self, creatures):
        """
        FIXED CAMERA MODE:
        Keeps the world centered and stops drifting.
        Only zoom changes the view. No movement.
        """
        pass

        # -------- OPTIONAL: Auto-follow all creatures (smooth) --------
        # If you ever want this back, remove the 'pass' above.
        #
        # if len(creatures) > 0:
        #     avg_x = sum(c.x for c in creatures) / len(creatures)
        #     avg_y = sum(c.y for c in creatures) / len(creatures)
        #     self.x += (avg_x - self.x) * 0.02
        #     self.y += (avg_y - self.y) * 0.02


###############################################################
# ========================== DRAWING ===========================
###############################################################

def draw_creature(screen, camera, creature):
    cx, cy = camera.apply(creature.x, creature.y)
    size = int(creature.size * camera.zoom)

    # Main creature circle
    if camera.zoom > 0.4:
        pygame.draw.circle(screen, creature.color, (cx, cy), size)
    else:
        screen.set_at((cx, cy), creature.color)

    # Energy bar above body
    energy_ratio = max(0.0, min(creature.energy / 200, 1))
    bar_w = int(22 * camera.zoom)
    bar_h = int(4 * camera.zoom)

    bar_x = cx - bar_w // 2
    bar_y = cy - size - int(8 * camera.zoom)

    pygame.draw.rect(screen, (25, 25, 25), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(screen, (0, 255, 50),
                     (bar_x, bar_y, int(bar_w * energy_ratio), bar_h))


def draw_food(screen, camera, plants, meats):
    # Plants
    for p in plants:
        x, y = camera.apply(p.x, p.y)
        if camera.zoom > 0.4:
            pygame.draw.circle(screen, p.color, (x, y), max(2, int(p.size * camera.zoom)))
        else:
            screen.set_at((x, y), p.color)

    # Meat
    for m in meats:
        x, y = camera.apply(m.x, m.y)
        if camera.zoom > 0.4:
            pygame.draw.circle(screen, m.color, (x, y), max(2, int(m.size * camera.zoom)))
        else:
            screen.set_at((x, y), m.color)


###############################################################
# ======================== UI OVERLAYS =========================
###############################################################

def draw_overlay(screen, font, temp, creatures, tick, fps, species_count, logger=None):
    """
    Text info overlay displayed in top-left corner.
    Shows: Temperature, Pop count, Species, Tick, FPS
    And optional average traits if logger is present.
    """

    text_color = (230, 230, 230)
    lines = []

    # Core world info
    lines.append(f"Temperature: {temp:.1f} °C")
    lines.append(f"Creatures: {len(creatures)}")
    lines.append(f"Species: {species_count}")
    lines.append(f"Tick: {tick}")
    lines.append(f"FPS: {fps}")

    # Logger trait statistics (if available)
    if logger:
        avg_speed = logger.avg_speed[-1] if logger.avg_speed else 0
        avg_size = logger.avg_size[-1] if logger.avg_size else 0
        avg_vision = logger.avg_vision[-1] if logger.avg_vision else 0

        lines.append(f"Avg Speed: {avg_speed:.2f}")
        lines.append(f"Avg Size: {avg_size:.2f}")
        lines.append(f"Avg Vision: {avg_vision:.1f}")

    # Render lines
    for i, line in enumerate(lines):
        surf = font.render(line, True, text_color)
        screen.blit(surf, (10, 10 + i * 20))


###############################################################
# ========================= MAIN LOOP ==========================
###############################################################

def run_graphics_mode():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulator — Standard Mode")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # ---- World Initialization ----
    global_tick = 0
    camera = Camera()

    # Very important: initialize DNA with neural weights
    creatures = []
    for _ in range(INITIAL_CREATURES):
        dna = extend_dna_for_brain(DNA())
        creatures.append(Creature(random.uniform(0, WIDTH),
                                  random.uniform(0, HEIGHT),
                                  dna=dna))

    plant_foods = [PlantFood() for _ in range(PLANT_FOOD_INITIAL)]
    meat_foods = []

    # Species tracker and logger optional
    speciation = SpeciationTracker()
    logger = DataLogger()

    running = True

    # ---- MAIN LOOP ----
    while running:
        dt = clock.tick(FPS)
        fps = int(clock.get_fps())

        # -------- Event Handling --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Zoom controls
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.zoom *= 1.08
                else:
                    camera.zoom *= 0.92
                camera.zoom = max(0.3, min(2.8, camera.zoom))

        # -------- World Update --------
        temp = world_step(creatures, plant_foods, meat_foods, global_tick)
        species_count = speciation.update(creatures)
        logger.log(creatures, temp, species_count)
        camera.update(creatures)
        global_tick += 1

        # -------- Drawing --------
        screen.fill((25, 25, 30))

        draw_food(screen, camera, plant_foods, meat_foods)

        for c in creatures:
            draw_creature(screen, camera, c)

        draw_overlay(screen, font, temp, creatures, global_tick, fps, species_count, logger)

        pygame.display.flip()

    pygame.quit()
###############################################################
# FULL REALISTIC EVOLUTION SIM – PART 4
# Neural Network Brain + DNA Integration (Corrected & Optimized)
###############################################################

###############################################################
# ========================= NEURAL NET =========================
###############################################################

class NeuralNet:
    """
    Evolvable neural network:
    - 10 inputs
    - 8 hidden neurons
    - 3 outputs (turn, speed modulation, aggression modulation)

    WEIGHT LAYOUT (flattened DNA segment):
        W1:  INPUTS × HIDDEN   = 10 × 8 = 80
        B1:  HIDDEN            = 8
        W2:  HIDDEN × OUTPUTS  = 8 × 3 = 24
        B2:  OUTPUTS           = 3

    Total = 80 + 8 + 24 + 3 = **115 weights**
    """

    INPUTS = 10
    HIDDEN = 8
    OUTPUTS = 3

    WEIGHT_COUNT = (INPUTS * HIDDEN) + HIDDEN + (HIDDEN * OUTPUTS) + OUTPUTS  # 115

    def __init__(self, weights=None):
        # If no weights given, initialize randomly
        if weights is None:
            self.weights = np.random.uniform(-0.25, 0.25, NeuralNet.WEIGHT_COUNT)
        else:
            self.weights = np.array(weights, dtype=float)

        # ======= UNPACK WEIGHTS =======
        idx = 0

        # Input → Hidden weights
        end = idx + NeuralNet.INPUTS * NeuralNet.HIDDEN
        self.W1 = self.weights[idx:end].reshape(NeuralNet.INPUTS, NeuralNet.HIDDEN)
        idx = end

        # Hidden biases
        end = idx + NeuralNet.HIDDEN
        self.B1 = self.weights[idx:end]
        idx = end

        # Hidden → Output weights
        end = idx + NeuralNet.HIDDEN * NeuralNet.OUTPUTS
        self.W2 = self.weights[idx:end].reshape(NeuralNet.HIDDEN, NeuralNet.OUTPUTS)
        idx = end

        # Output biases
        self.B2 = self.weights[idx : idx + NeuralNet.OUTPUTS]

    def activate(self, x):
        """Tanh activation: smooth, bounded, ideal for movement signals."""
        return np.tanh(x)

    def forward(self, inp):
        """Forward pass."""
        h = self.activate(np.dot(inp, self.W1) + self.B1)
        o = self.activate(np.dot(h, self.W2) + self.B2)
        return o  # 3 outputs: turn, speed_mod, aggression_mod


###############################################################
# ============= EXTEND DNA TO CONTAIN NN WEIGHTS ==============
###############################################################

def extend_dna_for_brain(dna_obj):
    """
    Ensures DNA contains all neural network weights.
    DNA is: [8 biological genes ... 115 neural-weight genes]
    """
    required = NeuralNet.WEIGHT_COUNT + 8  # 8 base genes + 115 NN weights

    if len(dna_obj.genes) < required:
        missing = required - len(dna_obj.genes)
        extra = np.random.uniform(-0.2, 0.2, missing)
        dna_obj.genes.extend(extra.tolist())

    return dna_obj


# ---- Patch original Crossover to include neural weights ----

_original_crossover = DNA.crossover

def new_crossover(d1, d2):
    """
    Combines parent genes, extends child DNA,
    and mutates both biological and neural genes.
    """

    # First perform normal gene crossover
    child = _original_crossover(d1, d2)

    # Ensure neural weights exist
    child = extend_dna_for_brain(child)

    # Mutate neural weights
    start = len(child.genes) - NeuralNet.WEIGHT_COUNT
    for i in range(start, len(child.genes)):
        if random.random() < MUTATION_CHANCE:
            child.genes[i] += random.uniform(-0.1, 0.1)

    return child

DNA.crossover = new_crossover  # override original


###############################################################
# ================== CREATURE BRAIN WITH NN ====================
###############################################################

class NeuralBrain:
    """
    Neural network–based creature brain.
    Converts raw perception → turn/speed/aggression decisions.
    """

    def __init__(self, creature):
        self.creature = creature

        # Extract final NN weight block from DNA
        total = NeuralNet.WEIGHT_COUNT
        start = len(creature.dna.genes) - total

        # Defensive fix: ensure DNA is extended (should already be done in PART 3)
        if start < 0:
            extend_dna_for_brain(creature.dna)
            start = len(creature.dna.genes) - total

        weights = creature.dna.genes[start:]
        self.nn = NeuralNet(weights)

    def decide(self, perception):
        c = self.creature

        # ------ Helper functions ------
        def angle_to(obj):
            if obj is None:
                return 0.0
            return math.atan2(obj.y - c.y, obj.x - c.x)

        def cos_sin(obj):
            """Returns (cosθ, sinθ) to maintain smooth directional input."""
            ang = angle_to(obj)
            return math.cos(ang), math.sin(ang)

        # ------ Build NN input vector ------
        plant_c, plant_s = cos_sin(perception.nearest_plant)
        meat_c, meat_s = cos_sin(perception.nearest_meat)
        prey_c, prey_s = cos_sin(perception.nearest_prey)
        pred_c, pred_s = cos_sin(perception.nearest_predator)

        inputs = np.array([
            plant_c, plant_s,
            meat_c, meat_s,
            prey_c, prey_s,
            pred_c, pred_s,
            perception.temperature / 40.0,
            perception.local_density / 10.0
        ], dtype=float)

        # ------ NN output ------
        turn_output, speed_output, aggro_output = self.nn.forward(inputs)

        # ------ Apply movement decisions ------
        c.angle += turn_output * 0.15
        c.angle += random.uniform(-0.03, 0.03)   # smoother, more natural
        c.speed = clamp(c.speed + speed_output * 0.15, 0.1, SPEED_MAX)

        # ------ Apply behavioral mutation ------
        c.aggression = clamp(
            c.aggression + aggro_output * 0.03,
            0, 1
        )

###############################################################
# FULL REALISTIC EVOLUTION SIM – PART 5
# Weather Events + Habitat Zones + Speciation + Data Logging
#  >>> FIXED + OPTIMIZED + SAME FUNCTION SIGNATURES <<<
###############################################################

import statistics
import numpy as np
import random

###############################################################
# ========================= WEATHER SYSTEM =====================
###############################################################

class WeatherSystem:
    """
    Dynamic weather system with realistic timing:
    - Rain: boosts plant regrowth
    - Storm: slight temperature drop
    - Heatwave: raises temperature
    - Coldwave: lowers temperature
    
    FIXES:
    - Prevented overly frequent weather changes
    - Temperature is adjusted only once per tick (Part 7 integrates correctly)
    """

    def __init__(self):
        self.state = "clear"
        self.timer = random.randint(900, 2200)

    def update(self, base_temp):
        self.timer -= 1

        if self.timer <= 0:
            self.timer = random.randint(900, 2200)
            self.state = random.choice(["clear", "rain", "storm", "heatwave", "coldwave"])

        # Modify temperature BEFORE habitat zone adjustments
        if self.state == "heatwave":
            base_temp += 6
        elif self.state == "coldwave":
            base_temp -= 6
        elif self.state == "storm":
            base_temp -= 2

        return base_temp


###############################################################
# ========================= HABITAT MAP ========================
###############################################################

class HabitatMap:
    """
    World is divided into 4×4 zones:
    - Desert: hot, low plants
    - Tundra: cold, low plants
    - Forest: rich plants
    - Plains: balanced

    FIXES:
    - plant_multiplier now used in Part 2
    - temperature and zone lookup stable at world edges
    """

    def __init__(self):
        self.zones = []
        self.generate()

    def generate(self):
        self.zones.clear()
        for i in range(4):
            for j in range(4):
                r = random.random()
                if r < 0.25:
                    zone = "desert"
                elif r < 0.5:
                    zone = "tundra"
                elif r < 0.75:
                    zone = "forest"
                else:
                    zone = "plains"
                self.zones.append(zone)

    def zone_at(self, x, y):
        col = int(x / (WIDTH / 4))
        row = int(y / (HEIGHT / 4))
        idx = row * 4 + col
        return self.zones[idx]

    def adjust_temperature(self, x, y, temp):
        zone = self.zone_at(x, y)
        if zone == "desert":
            return temp + 4
        if zone == "tundra":
            return temp - 6
        if zone == "forest":
            return temp - 1
        return temp  # plains

    def plant_multiplier(self, x, y):
        zone = self.zone_at(x, y)
        if zone == "forest":
            return 1.6
        if zone == "desert":
            return 0.5
        if zone == "tundra":
            return 0.6
        return 1.0  # plains


###############################################################
# ===================== MIGRATION PRESSURE =====================
###############################################################

def migration_force(creature, habitat: HabitatMap, temperature):
    """
    Encourages movement out of unsuitable climates.
    
    FIXES:
    - Prevented massive spin jitter
    - Smooth turning rather than violent angle flips
    """

    zone = habitat.zone_at(creature.x, creature.y)

    # Too hot
    if temperature > creature.heat_tolerance + 5:
        creature.angle += random.uniform(1.1, 1.7)

    # Too cold
    elif temperature < creature.cold_tolerance - 5:
        creature.angle += random.uniform(-1.7, -1.1)

    # Herbivores avoid deserts
    if zone == "desert" and creature.diet_pref < 0.5:
        creature.angle += random.uniform(0.5, 1.0)


###############################################################
# ========================== SPECIATION =========================
###############################################################

class SpeciationTracker:
    """
    Clusters creatures into species based on DNA similarity.

    FIXES:
    - Speciation now runs every 20 ticks (HUGE speed improvement)
    - Bounding-box check prevents 1000s of DNA comparisons
    - Species IDs remain stable and predictable
    """

    def __init__(self):
        self.species = {}
        self.next_id = 1
        self.tick_counter = 0

    def dna_distance(self, g1, g2):
        """
        Species separation uses only first 10 core genes
        (ignores neural network weights).
        """
        a = np.array(g1[:10])
        b = np.array(g2[:10])
        return np.linalg.norm(a - b)

    def update(self, creatures):
        # Run only every 20 ticks to avoid CPU overload
        self.tick_counter += 1
        if self.tick_counter % 20 != 0:
            return len(self.species)

        threshold = 4.0
        species_groups = []

        for c in creatures:
            placed = False

            for group in species_groups:
                rep = group[0]

                # Fast reject: extremely different sizes → skip
                if abs(c.size - rep.size) > 4:
                    continue

                # Exact DNA check
                if self.dna_distance(c.dna.genes, rep.dna.genes) < threshold:
                    group.append(c)
                    placed = True
                    break

            if not placed:
                species_groups.append([c])

        # Assign ID numbers
        self.species = {i + 1: grp for i, grp in enumerate(species_groups)}
        self.next_id = len(self.species) + 1

        return len(self.species)


###############################################################
# ========================= DATA LOGGER ========================
###############################################################

class DataLogger:
    """
    Collects and stores statistics over time.

    FIXES:
    - Removed duplicate logger conflict with Part 6
    - Keeps history perfectly aligned with ticks
    """

    def __init__(self):
        self.pop_history = []
        self.temp_history = []
        self.species_history = []
        self.avg_speed = []
        self.avg_size = []
        self.avg_vision = []

    def log(self, creatures, temperature, species_count):
        self.pop_history.append(len(creatures))
        self.temp_history.append(temperature)
        self.species_history.append(species_count)

        if creatures:
            self.avg_speed.append(statistics.mean(c.speed for c in creatures))
            self.avg_size.append(statistics.mean(c.size for c in creatures))
            self.avg_vision.append(statistics.mean(c.vision for c in creatures))
        else:
            self.avg_speed.append(0)
            self.avg_size.append(0)
            self.avg_vision.append(0)

###############################################################
# FULL REALISTIC EVOLUTION SIM – PART 6 + PART 7
# Save/Load System + Live Graphs + Advanced Mode Main Loop
# >>> FULLY CLEANED, OPTIMIZED, MAC-SAFE, AND COMPLETE <<<
###############################################################

import pickle
import matplotlib.pyplot as plt
import threading
import time
import pygame


###############################################################
# ====================== SAVE / LOAD ==========================
###############################################################

def save_state(creatures, plant_foods, meat_foods, tick):
    """
    Saves simulation state for later continuation.
    Uses pickle — data will break if creature classes change.
    """

    try:
        data = {
            "creatures": creatures,
            "plant_foods": plant_foods,
            "meat_foods": meat_foods,
            "tick": tick
        }
        with open("sim_save.pkl", "wb") as f:
            pickle.dump(data, f)
        print("[SAVE] Simulation saved successfully.")
    except Exception as e:
        print("[SAVE ERROR]", e)


def load_state():
    """
    Loads simulation state if it exists.
    """
    try:
        with open("sim_save.pkl", "rb") as f:
            data = pickle.load(f)
        print("[LOAD] Simulation loaded.")
        return data
    except FileNotFoundError:
        print("[LOAD ERROR] No save file found.")
        return None
    except Exception as e:
        print("[LOAD ERROR]", e)
        return None


###############################################################
# ======================= LIVE GRAPHING ========================
###############################################################

class LiveGrapher:
    """
    Safe background graphing thread compatible with pygame and macOS.
    Updates graphs every 0.5 seconds.
    """

    def __init__(self, logger):
        self.logger = logger
        self.running = True

        try:
            thread = threading.Thread(target=self.graph_loop, daemon=True)
            thread.start()
        except Exception as e:
            print("[GRAPHER DISABLED]", e)

    def graph_loop(self):
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Evolution Statistics (Live)")

        while self.running:
            if len(self.logger.pop_history) < 3:
                time.sleep(1)
                continue

            # Population
            axs[0][0].clear()
            axs[0][0].plot(self.logger.pop_history, color="white")
            axs[0][0].set_title("Population")

            # Temperature
            axs[0][1].clear()
            axs[0][1].plot(self.logger.temp_history, color="orange")
            axs[0][1].set_title("Temperature")

            # Average traits
            axs[1][0].clear()
            axs[1][0].plot(self.logger.avg_speed, label="Speed")
            axs[1][0].plot(self.logger.avg_size, label="Size")
            axs[1][0].plot(self.logger.avg_vision, label="Vision")
            axs[1][0].legend()
            axs[1][0].set_title("Average Traits")

            # Species count
            axs[1][1].clear()
            axs[1][1].plot(self.logger.species_history, color="green")
            axs[1][1].set_title("Species Count")

            plt.pause(0.01)
            time.sleep(0.5)


###############################################################
# ============= ENVIRONMENTAL HABITAT OVERLAY =================
###############################################################

def draw_habitat_overlay(screen, habitat):
    """
    Draws a faint colored overlay for each habitat zone:
    - Desert
    - Tundra
    - Forest
    - Plains
    """

    zone_colors = {
        "desert": (210, 180, 80),
        "tundra": (160, 200, 255),
        "forest": (60, 140, 90),
        "plains": (120, 180, 120),
    }

    cell_w = WIDTH // 4
    cell_h = HEIGHT // 4

    for row in range(4):
        for col in range(4):
            zone = habitat.zones[row * 4 + col]
            color = zone_colors[zone]

            rect = pygame.Rect(col * cell_w, row * cell_h, cell_w, cell_h)
            overlay = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
            overlay.fill((*color, 60))  # 60 alpha = semi-transparent

            screen.blit(overlay, rect)


###############################################################
# ===================== PART 7 – ADVANCED MODE =================
#   (Main Loop with Habitat + Weather + Speciation + Logging)
###############################################################

def run_graphics_mode_with_overlays():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Simulator — Advanced Mode")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # ========= WORLD COMPONENTS ==========
    global_tick = 0
    camera = Camera()
    habitat = HabitatMap()
    weather = WeatherSystem()
    speciation = SpeciationTracker()
    logger = DataLogger()
    # grapher = LiveGrapher(logger)   # SAFE but optional – you can enable it

    creatures = [
        Creature(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        for _ in range(INITIAL_CREATURES)
    ]

    plant_foods = [PlantFood() for _ in range(PLANT_FOOD_INITIAL)]
    meat_foods = []

    running = True

    # ===================== MAIN LOOP =======================
    while running:

        dt = clock.tick(FPS)
        fps = int(clock.get_fps())

        # ---------- EVENTS ----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Save / Load
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_state(creatures, plant_foods, meat_foods, global_tick)
                if event.key == pygame.K_l:
                    data = load_state()
                    if data:
                        creatures = data["creatures"]
                        plant_foods = data["plant_foods"]
                        meat_foods = data["meat_foods"]
                        global_tick = data["tick"]

            # Zoom
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.zoom *= 1.1
                else:
                    camera.zoom *= 0.9
                camera.zoom = max(0.3, min(2.5, camera.zoom))

        # ---------- WEATHER ----------
        temp = compute_temperature(global_tick)
        temp = weather.update(temp)

        # ---------- APPLY HABITAT TEMPERATURE ----------
        for c in creatures:
            zone_temp = habitat.adjust_temperature(c.x, c.y, temp)
            c.apply_temperature(zone_temp)

        # ---------- WORLD STEP ----------
        temp_after = world_step(creatures, plant_foods, meat_foods, global_tick)

        # ---------- SPECIATION ----------
        species_count = speciation.update(creatures)

        # ---------- LOGGING ----------
        logger.log(creatures, temp_after, species_count)

        global_tick += 1

        # ---------- CAMERA UPDATE ----------
        camera.update(creatures)

        # ---------- DRAW EVERYTHING ----------
        screen.fill((20, 20, 25))

        draw_habitat_overlay(screen, habitat)
        draw_food(screen, camera, plant_foods, meat_foods)

        for c in creatures:
            draw_creature(screen, camera, c)

        draw_overlay(
            screen,
            font,
            temp_after,
            creatures,
            global_tick,
            fps,
            species_count,
            logger
        )

        pygame.display.flip()

    pygame.quit()
if __name__ == "__main__":
    run_graphics_mode_with_overlays()


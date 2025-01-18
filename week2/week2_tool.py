import math

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def dist(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx**2 +dy**2)

class line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
    def slope(self):
        distx = self.p2.x - self.p1.x
        disty = self.p2.y - self.p1.y
        if distx == 0:
            return None
        return disty/distx
    
    def parallel(self, other):
        m1 = self.slope()
        m2 = other.slope()
        if m1 is None and m2 is None:
            return True        
        if m1 is None or m2 is None:
            return False
        
        return math.isclose(m1, m2)
    
    def perpendicular(self, other):
        m1 = self.slope()
        m2 = other.slope()
        if m1 is None and m2 == 0:
            return True
        if m2 is None and m1 == 0:
            return True
        if (m1 is not None) and (m2 is not None):
            return math.isclose(m1 * m2, -1)
        
        return False
    
class circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    
    def area(self):
        return math.pi*(self.radius**2)
    
    def intersects(self, other):
        center_dist = self.center.dist(other.center)
        
        radius_sum = self.radius + other.radius
        
        return center_dist < radius_sum

class polygon:
    def __init__(self, points):
        self.points = points
    
    def perimeter(self):
        n = len(self.points)
        total = 0
        for i in range(n):
            p1 = self.points[i]
            p2 = self.points[(i+1) % n]
            total = total + p1.dist(p2)
        return total      
    
class Slime:
    def __init__(self, name, x, y, vx, vy, hp=10):
        self.name = name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.hp = hp
        self.alive = True
    
    def move(self):
        if self.alive:
            self.x += self.vx
            self.y += self.vy    

    def take_damage(self, damage):
        if self.alive:
            self.hp -= damage
            if self.hp <= 0:
                self.hp = 0
                self.alive = False

    def dist_to_tower(self, tower):
        dx = tower.x - self.x
        dy = tower.y - self.y
        return math.sqrt(dx*dx + dy*dy)

class Tower:
    def __init__(self, name, x, y, attack, range):
        self.name = name
        self.x = x
        self.y = y
        self.attack = attack
        self.range = range 

    def attack_slime(self, slime_list):
        for i in slime_list:
            if i.alive:
                dist = i.dist_to_tower(self)
                if dist <= self.range:
                    i.take_damage(self.attack)
class BasicTower(Tower):
    def __init__(self, name, x, y):
        super().__init__(name, x, y, 1, 2)

class AdvancedTower(Tower):
    def __init__(self, name, x, y):
        super().__init__(name, x, y, 2, 4)
from week2_tool import point,line,circle, polygon, Slime, BasicTower, AdvancedTower

#task 1
A1 = point(-6, 1)
A2 = point(2, 4)
lineA = line(A1, A2)

B1 = point(-6, -1)
B2 = point(2, 2)
lineB = line(B1, B2)

C1 = point(-1, -6)
C2 = point(-4, -4)
lineC = line(C1, C2)

circleA = circle(point(6, 3), 2)

circleB = circle(point(8, 1), 1)

polyA_points = [
    point(-1, -2),
    point(2, 0),
    point(5, -1),
    point(4, -4),
]
polygonA = polygon(polyA_points)

print("\n=== Task 1  ===")
print("1. Are line A and line B parallel?")
print("   A:", "Yes" if lineA.parallel(lineB) else "No")


print("2. Are line C and line A perpendicular?")
print("   A:", "Yes" if lineC.perpendicular(lineA) else "No")

areaA = circleA.area()
print(f"3. Circle A's area = {areaA:.4f}")

print("4. Do Circle A and Circle B intersect? ")
print("   A:", "Yes" if circleA.intersects(circleB) else "No")

perimeterA = polygonA.perimeter()
print(f"5. perimeter of Polygon A = {perimeterA:.4f}")

#Task 2
slimes = [
    Slime("Slime_1", -10, 2, 2, -1, 10),
    Slime("Slime_2", -8, 0, 3, 1, 10),
    Slime("Slime_3", -9, -1, 3, 0, 10)
]

towers = [
    BasicTower("T1", -3, 2),
    BasicTower("T2", -1, -2),
    BasicTower("T3", 4, 2),
    BasicTower("T4", 7, 0),
    AdvancedTower("A1", 1, 1),
    AdvancedTower("A2", 4, -3)
]

number_truns = 10
for turn in range(1, number_truns + 1):
    for s in slimes:
        s.move()
    for t in towers:
        t.attack_slime(slimes)

print("\n=== Task 2  ===")

print("\n=== After 10 turns, slimes' final status ===")
for s in slimes:
    status = "alive~" if s.alive else "dead!"
    print(f"[{s.name}] position=({s.x},{s.y}), hp={s.hp}, {s.name} is {status}.")

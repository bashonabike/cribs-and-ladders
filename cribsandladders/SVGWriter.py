import svgwrite

# Example data
coordinate_points = [(50, 50), (150, 150), (250, 50)]
vectors = [((50, 50), (100, 100)), ((150, 150), (200, 50)), ((250, 50), (300, 150))]

# Create an SVG drawing object
dwg = svgwrite.Drawing('output.svg', profile='tiny')

# Create layers
points_layer = dwg.add(dwg.g(id='points_layer'))
vectors_layer = dwg.add(dwg.g(id='vectors_layer'))

# Add coordinate points to the points layer
for point in coordinate_points:
    points_layer.add(dwg.circle(center=point, r=5, fill='red'))

# Add vectors to the vectors layer
for start, end in vectors:
    vectors_layer.add(dwg.line(start=start, end=end, stroke='blue', stroke_width=2))

# Save the drawing to a file
dwg.save()
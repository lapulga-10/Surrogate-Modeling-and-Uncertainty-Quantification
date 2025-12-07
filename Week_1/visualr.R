library(rgl)



rm(list = ls())

load("Finite_Element_Mesh.RData")

print(ls())

head(coordinates)
head(edge)
head(membrane)
head(support)
head(truss)

dim(coordinates)[1] 

dim(edge)[1] +
dim(membrane)[1] +
dim(support)[1] +
dim(truss)[1]



# --- Prepare mesh data ---
points <- coordinates[, 2:4]
triangles <- membrane[, 2:4]


# --- Open a 3D window ---
open3d()
bg3d(color = "white")
view3d(theta = 30, phi = 20, zoom = 0.8)




triangles3d(
  x = points[triangles, 1],
  y = points[triangles, 2],
  z = points[triangles, 3],
  color = "blue",
  alpha = 0.6
)



# --- Function to draw line elements (edges, supports, trusses) ---
draw_lines <- function(line_data, color, lwd = 2) {
  for (i in 1:nrow(line_data)) {
    p1 <- points[line_data[i, 1], ]
    p2 <- points[line_data[i, 2], ]
    segments3d(
      rbind(p1, p2),
      color = color,
      lwd = lwd
    )
  }
}


# --- Add edges, supports, and trusses ---
draw_lines(edge[, 2:3], color = "darkred", lwd = 2)
draw_lines(support[, 2:3], color = "green", lwd = 2)
draw_lines(truss[, 2:3], color = "orange", lwd = 2)


# --- Add legend ---
legend3d(
  "topright",
  legend = c("Edges", "Supports", "Trusses"),
  col = c("darkred", "green", "orange"),
  lwd = 3,
  cex = 1
)


import torch
import matplotlib.pyplot as plt


def pythagoras_tree(depth, angle=torch.pi / 6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_tree(x, y, length, angle, depth):
        if depth == 0:
            return torch.tensor([])

        # Calculate end point of the branch
        x_end = x + length * torch.cos(angle)
        y_end = y + length * torch.sin(angle)

        # Create current branch
        branch = torch.tensor([[x, y, x_end.item(), y_end.item()]])

        # Calculate new length and angles for sub-branches
        new_length = length / (3 ** 0.5)  # Divide by sqrt(3) for 3 branches
        angles = torch.tensor([-angle, 0, angle]) + angle

        # Recursively generate sub-branches
        sub_branches = [generate_tree(x_end, y_end, new_length, angle + a, depth - 1) for a in angles]

        return torch.cat([branch] + sub_branches)

    # Initial setup
    start_x = 0
    start_y = 0
    start_length = 1
    start_angle = torch.tensor(torch.pi / 2)  # Start vertically

    # Generate the tree
    tree = generate_tree(start_x, start_y, start_length, start_angle, depth)

    return tree.to(device)


# Set the depth of the tree
depth = 8

# Generate the tree
tree = pythagoras_tree(depth)

# Plot the tree
plt.figure(figsize=(10, 10))
for branch in tree:
    plt.plot(branch[:2], branch[2:], 'k-')
plt.axis('equal')
plt.axis('off')
plt.title(f"3-Branched Pythagoras Tree (Depth: {depth})")
plt.show()

# Calculate and print log2(3)
log2_3 = torch.log2(torch.tensor(3.0))
print(f"log2(3) ≈ {log2_3.item():.4f}")
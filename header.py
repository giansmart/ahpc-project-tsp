import csv

def write_to_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])  # Write header row


filename = 'output.csv'
write_to_csv(filename)
print(f'Data written to {filename}')

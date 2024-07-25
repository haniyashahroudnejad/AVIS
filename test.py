def log_to_markdown(file_path, log_entries):
    with open(file_path, 'w') as f:
        for entry in log_entries:
            color = entry.get('color', 'black')
            text = entry.get('text', '')
            f.write(f'<p style="color:{color};">{text}</p>\n')

log_entries = [
    {'color': 'red', 'text': 'Error: Something went wrong.'},
    {'color': 'green', 'text': 'Success: Operation completed successfully.'},
    {'color': 'blue', 'text': 'Info: Here is some information.'}
]

log_to_markdown('log.md', log_entries)

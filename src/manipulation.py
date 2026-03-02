# 123456|E1,5
# 123456|D2
# 123456|I2,3
# 123456|F

def manipulate(s: str, command: str) -> str:
    command, args = extract_command_and_args(command)
    match (command, args):
        case ("E", [index, char]):
            index = int(index)
            return s[:index] + char + s[index + 1 :]
        case ("D", [index]):
            index = int(index)
            return s[:index] + s[index + 1 :]
        case ("I", [index, char]):
            index = int(index)
            return s[:index] + char + s[index:]
        case _:
            # print(f"Unknown command: {command} {args}")
            return s

def get_opposite_command(s: str, command: str) -> str:
    command, args = extract_command_and_args(command)
    match (command, args):
        case ("E", [index, char]):
            index = int(index)
            original_char = s[index]
            return "E" + str(index) + "," + original_char
        case ("D", [index]):
            index = int(index)
            original_char = s[index]
            return "I" + str(index) + "," + original_char
        case ("I", [index, char]):
            index = int(index)
            return "D" + str(index)
        case _:
            # print(f"Unknown command: {command} {args}")
            return s
    
def extract_command_and_args(command: str) -> tuple[str, list[str]]:
    command = command.strip()
    operation = command[0]
    args = command[1:].split(",")
    return operation, args
import os
import json
import torch
from src.datautils import read_jsonl
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

CODE_SMELL_SYSTEM_PROMPT = """You are an expert code reviewer and your task is to look at code and detect any issues also called code smells. A code smell is a surface indication that usually corresponds to a deeper problem in the system. Some example potentially smelly code snippets and the type of smell they contain (if any) are shown below along with explanations of how they exhibit the smell. Look at them and learn to detect code smells.

Note that only these code smells exist, so your answer should be one of these:
1. Duplicated Code
2. Long Method 
3. Large Class
4. Long Parameter List
5. Divergent Change
6. Shotgun Surgery
7. Feature Envy
8. Data Clumps
9. Primitive Obsession
10. Switch Statements
11. Parallel Inheritance Hierarchies
12. Lazy Class
13. Speculative Generality
14. Temporary Field
15. Message Chains
16. Middle Man
17. Inappropriate Intimacy
18. Alternative Classes with Different Interfaces
19. Incomplete Library Class
20. Data Class 
21. Refused Bequest
22. No Smell
"""

CODE_SMELL_SIMPLIFIED_DETECTION_PROMPT = """Code: 
def calculate_area_of_square(side_length):
    return side_length * side_length

def calculate_area_of_rectangle(length, width):
    return length * width

Smell: Duplicated Code

Explanation: Both functions calculate_area_of_square and calculate_area_of_rectangle share a similar computation for calculating area, leading to duplicated code.

Code:
def calculate_employee_salary(employee):
    base_salary = employee.base_salary
    bonuses = employee.bonuses
    deductions = employee.deductions
    tax_rate = employee.tax_rate
    
    gross_salary = base_salary + bonuses - deductions
    net_salary = gross_salary * (1 - tax_rate)
    
    return net_salary

Smell: Long Method

Explanation: The calculate_employee_salary method is lengthy and performs multiple operations, which reduces readability and maintainability.

Code:
class Customer:
    def __init__(self, name, age, email, address, phone_number, credit_card_info):
        self.name = name
        self.age = age
        self.email = email
        self.address = address
        self.phone_number = phone_number
        self.credit_card_info = credit_card_info

    def place_order(self, order):
        # Method implementation here
        pass
    
    def cancel_order(self, order):
        # Method implementation here
        pass
    
    # Other methods...

Smell: Large Class

Explanation: The Customer class has many attributes and methods, which may indicate that it is doing too much and violating the Single Responsibility Principle.

Code:
def send_notification(subject, body, recipient_email, cc_emails, bcc_emails, attachment, urgency):
    # Function implementation here
    pass

Smell: Long Parameter List

Explanation: The send_notification function has a long list of parameters, which can make it hard to use and understand.

Code:
class ShoppingCart:
    def add_item(self, item):
        # Method implementation here
        pass
    
    def remove_item(self, item):
        # Method implementation here
        pass
    
    def calculate_total(self):
        # Method implementation here
        pass
    
    def apply_discount(self, discount):
        # Method implementation here
        pass

class PaymentProcessor:
    def process_payment(self, amount, payment_method):
        # Method implementation here
        pass
    
    def refund_payment(self, amount, payment_method):
        # Method implementation here
        pass

Smell: Divergent Change

Explanation: Changes to shopping cart functionality require modifications across multiple methods within the ShoppingCart class and possibly across different classes such as PaymentProcessor, indicating divergent changes.

Code:
class UserManager:
    def promote_user(self, user):
        # Method implementation here
        pass
    
    def demote_user(self, user):
        # Method implementation here
        pass

class EmailSender:
    def send_promotion_email(self, user):
        # Method implementation here
        pass
    
    def send_demotion_email(self, user):
        # Method implementation here
        pass

Smell: Shotgun Surgery

Explanation: Making changes to user promotion or demotion functionality requires modifications in both the UserManager and EmailSender classes, indicating shotgun surgery.

Code:
class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items
    
    def calculate_total_price(self):
        total_price = 0
        for item in self.items:
            total_price += item.price
        return total_price

Smell: Feature Envy

Explanation: The calculate_total_price method seems more interested in the item attributes than the Order attributes, suggesting that it may belong more naturally to the Item class, indicating feature envy.

Code:
class Order:
    def __init__(self, item_name, item_price, quantity):
        self.item_name = item_name
        self.item_price = item_price
        self.quantity = quantity

    def calculate_total_price(self):
        return self.item_price * self.quantity

Smell: Primitive Obsession

Explanation: The Order class uses primitive data types (item_name, item_price, quantity) instead of more meaningful abstractions, indicating primitive obsession.

Code: 
def calculate_discount(item_type, price):
    if item_type == "book":
        return price * 0.1
    elif item_type == "clothing":
        return price * 0.2
    elif item_type == "electronics":
        return price * 0.15
    else:
        return 0

Smell: Switch Statements

Explanation: The calculate_discount function uses switch statements based on the item_type, which can be a sign of procedural code and violation of the Open/Closed Principle.

Explanation: When a class hierarchy mirrors another hierarchy instead of being unified, it's called a parallel inheritance hierarchy. In this example, the Animal hierarchy mirrors another hierarchy (e.g., Dog, Cat, Fish), which can lead to maintenance issues.

Code:
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

class Department:
    def __init__(self, name, employees):
        self.name = name
        self.employees = employees

Smell: Lazy Class

Explanation: The Employee class doesn't have enough behavior to justify its existence. It only contains basic data and lacks meaningful functionality, making it a lazy class.

Code:
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return 3.14 * self.radius * self.radius

    def calculate_circumference(self):
        return 2 * 3.14 * self.radius

Smell: No smell

Explanation: This class represents a circle with methods to calculate its area and circumference. It adheres to the Single Responsibility Principle by focusing on a single concept (circle) and doesn't exhibit any of the common code smells such as duplicated code, long methods, or primitive obsession. It's concise, readable, and well-organized.

Code:
{after_code}

Smell:"""

CODE_SMELL_DETECTION_PROMPT = """Code: 
def calculate_area_of_square(side_length):
    return side_length * side_length

def calculate_area_of_rectangle(length, width):
    return length * width

Smell: Duplicated Code

Explanation: Both functions calculate_area_of_square and calculate_area_of_rectangle share a similar computation for calculating area, leading to duplicated code.

Code:
def calculate_employee_salary(employee):
    base_salary = employee.base_salary
    bonuses = employee.bonuses
    deductions = employee.deductions
    tax_rate = employee.tax_rate
    
    gross_salary = base_salary + bonuses - deductions
    net_salary = gross_salary * (1 - tax_rate)
    
    return net_salary

Smell: Long Method

Explanation: The calculate_employee_salary method is lengthy and performs multiple operations, which reduces readability and maintainability.

Code:
class Customer:
    def __init__(self, name, age, email, address, phone_number, credit_card_info):
        self.name = name
        self.age = age
        self.email = email
        self.address = address
        self.phone_number = phone_number
        self.credit_card_info = credit_card_info

    def place_order(self, order):
        # Method implementation here
        pass
    
    def cancel_order(self, order):
        # Method implementation here
        pass
    
    # Other methods...

Smell: Large Class

Explanation: The Customer class has many attributes and methods, which may indicate that it is doing too much and violating the Single Responsibility Principle.

Code:
def send_notification(subject, body, recipient_email, cc_emails, bcc_emails, attachment, urgency):
    # Function implementation here
    pass

Smell: Long Parameter List

Explanation: The send_notification function has a long list of parameters, which can make it hard to use and understand.

Code:
class ShoppingCart:
    def add_item(self, item):
        # Method implementation here
        pass
    
    def remove_item(self, item):
        # Method implementation here
        pass
    
    def calculate_total(self):
        # Method implementation here
        pass
    
    def apply_discount(self, discount):
        # Method implementation here
        pass

class PaymentProcessor:
    def process_payment(self, amount, payment_method):
        # Method implementation here
        pass
    
    def refund_payment(self, amount, payment_method):
        # Method implementation here
        pass

Smell: Divergent Change

Explanation: Changes to shopping cart functionality require modifications across multiple methods within the ShoppingCart class and possibly across different classes such as PaymentProcessor, indicating divergent changes.

Code:
class UserManager:
    def promote_user(self, user):
        # Method implementation here
        pass
    
    def demote_user(self, user):
        # Method implementation here
        pass

class EmailSender:
    def send_promotion_email(self, user):
        # Method implementation here
        pass
    
    def send_demotion_email(self, user):
        # Method implementation here
        pass

Smell: Shotgun Surgery

Explanation: Making changes to user promotion or demotion functionality requires modifications in both the UserManager and EmailSender classes, indicating shotgun surgery.

Code:
class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items
    
    def calculate_total_price(self):
        total_price = 0
        for item in self.items:
            total_price += item.price
        return total_price

Smell: Feature Envy

Explanation: The calculate_total_price method seems more interested in the item attributes than the Order attributes, suggesting that it may belong more naturally to the Item class, indicating feature envy.

Code:
class Address:
    def __init__(self, street, city, state, zipcode):
        self.street = street
        self.city = city
        self.state = state
        self.zipcode = zipcode

class Customer:
    def __init__(self, name, age, email, address):
        self.name = name
        self.age = age
        self.email = email
        self.address = address

Smell: Data Clumps

Explanation: The Customer class uses the Address class to group related address attributes together, indicating data clumps.

Code:
class Order:
    def __init__(self, item_name, item_price, quantity):
        self.item_name = item_name
        self.item_price = item_price
        self.quantity = quantity

    def calculate_total_price(self):
        return self.item_price * self.quantity

Smell: Primitive Obsession

Explanation: The Order class uses primitive data types (item_name, item_price, quantity) instead of more meaningful abstractions, indicating primitive obsession.

Code: 
def calculate_discount(item_type, price):
    if item_type == "book":
        return price * 0.1
    elif item_type == "clothing":
        return price * 0.2
    elif item_type == "electronics":
        return price * 0.15
    else:
        return 0

Smell: Switch Statements

Explanation: The calculate_discount function uses switch statements based on the item_type, which can be a sign of procedural code and violation of the Open/Closed Principle.

Code:
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"
    
class Fish(Animal):
    def speak(self):
        return "" # Fish don't speak, but we have to override the method due to hierarchy.

Smell: Parallel Inheritance Hierarchies

Explanation: When a class hierarchy mirrors another hierarchy instead of being unified, it's called a parallel inheritance hierarchy. In this example, the Animal hierarchy mirrors another hierarchy (e.g., Dog, Cat, Fish), which can lead to maintenance issues.

Code:
class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary

class Department:
    def __init__(self, name, employees):
        self.name = name
        self.employees = employees

Smell: Lazy Class

Explanation: The Employee class doesn't have enough behavior to justify its existence. It only contains basic data and lacks meaningful functionality, making it a lazy class.

Code:
class PaymentGateway:
    def make_payment(self, amount, payment_method):
        pass

    def process_refund(self, amount, payment_method):
        pass

    def calculate_exchange_rate(self, currency_from, currency_to):
        pass

Smell: Speculative Generality

Explanation: The PaymentGateway class includes methods that may not currently be needed. It's an example of speculative generality, where functionality is provided in anticipation of future requirements that may never materialize.

Code:
class Order:
    def __init__(self, items):
        self.items = items

    def calculate_total_price(self):
        total_price = 0
        for item in self.items:
            total_price += item.price
        return total_price

class Item:
    def __init__(self, name, price):
        self.name = name
        self.price = price

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def calculate_total_price(self):
        total_price = 0
        for item in self.items:
            total_price += item.price
        return total_price

Smell: Temporary Field

Explanation: The Order class temporarily stores a list of items (self.items) but doesn't use it to fulfill its responsibilities. It's a sign of temporary field smell.

Code:
class Address:
    def __init__(self, street, city, country):
        self.street = street
        self.city = city
        self.country = country

class Customer:
    def __init__(self, name, address):
        self.name = name
        self.address = address

class Order:
    def __init__(self, customer):
        self.customer = customer

order = Order(Customer("John", Address("123 Main St", "Springfield", "USA")))
country = order.customer.address.country

Smell: Message Chains

Explanation: Message chains occur when client code navigates through multiple objects to get the required information. In this example, accessing the country of the customer's address involves traversing through multiple objects (order.customer.address.country).

Code:
class Department:
    def __init__(self, manager):
        self.manager = manager

    def get_manager_name(self):
        return self.manager.name

Smell: Middle Man

Explanation: The Department class acts as a middleman between clients and the Manager class, forwarding requests to the manager. This unnecessary delegation makes the class a middle man.

Code:
class Employee:
    def __init__(self, name, department):
        self.name = name
        self.department = department

class Department:
    def __init__(self, name):
        self.name = name
        self.employees = []

    def add_employee(self, employee):
        self.employees.append(employee)

Smell: Inappropriate Intimacy

Explanation: The Employee class holds a reference to the Department class (self.department), and the Department class holds a list of employees (self.employees). This tight coupling between the two classes represents inappropriate intimacy.

Code:
class Square:
    def __init__(self, side_length):
        self.side_length = side_length

    def calculate_area(self):
        return self.side_length * self.side_length

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def calculate_area(self):
        return self.length * self.width

Smell: Alternative Classes with Different Interfaces:

Explanation: Both Square and Rectangle classes represent geometric shapes, but they have different interfaces (side_length vs. length and width). This inconsistency in interfaces can lead to confusion and errors.

Code:
# Let's say there's a library class provided by a third-party without some necessary functionality

class ThirdPartyLibraryClass:
    def method1(self):
        pass

    # This class is incomplete and lacks necessary functionality, but we cannot modify it

Smell: Incomplete Library Class

Explanation: The ThirdPartyLibraryClass lacks some necessary functionality required by the client code. This forces client code to implement additional functionality, resulting in incomplete library class smell.

Code:
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

Smell: Data Class

Explanation: The Rectangle class is a simple container for data without any additional behavior. It's an example of a data class, which is not inherently a code smell but can become one if the class grows in complexity without adding behavior.

Code:
class Vehicle:
    def start_engine(self):
        pass

class Car(Vehicle):
    def start_engine(self):
        # Car engine starting process
        pass

class Bicycle(Vehicle):
    pass

Smell: Refused Bequest

Explanation: The Bicycle class inherits from Vehicle but refuses the inherited start_engine method. This refusal of inherited behavior indicates a design flaw known as refused bequest.

Code:
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return 3.14 * self.radius * self.radius

    def calculate_circumference(self):
        return 2 * 3.14 * self.radius

Smell: No smell

Explanation: This class represents a circle with methods to calculate its area and circumference. It adheres to the Single Responsibility Principle by focusing on a single concept (circle) and doesn't exhibit any of the common code smells such as duplicated code, long methods, or primitive obsession. It's concise, readable, and well-organized.

Code:
{after_code}

Smell:"""


class LLMCodeSmellDetector:
    def __init__(self, model_id, quantization: bool=True, quantization_type: int=4):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # if quantization:
        #     if quantization_type == 4:
        #         self.quantization_config = quantization_config = BitsAndBytesConfig(
        #             load_in_4bit=True,
        #             bnb_4bit_compute_dtype=torch.float16
        #         )
        #         self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        #     if quantization_type == 8:
        #         self.quantization_config = quantization_config = BitsAndBytesConfig(
        #             load_in_8bit=True,
        #             bnb_8bit_compute_dtype=torch.float16
        #         )
        #         self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
        # else:
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        if torch.cuda.is_available(): self.model.cuda()
    
    def __call__(self, code):
        # print("LLM call")
        # start = time.time()
        messages = [
            {
                "role": "user", 
                "content": CODE_SMELL_SYSTEM_PROMPT+"\n"+CODE_SMELL_SIMPLIFIED_DETECTION_PROMPT.format(
                    after_code=code,
                )
            }
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        # tokenized_chat = tokenized_chat 
        outputs = self.model.generate(
            tokenized_chat.to(self.model.device),
            max_new_tokens=2048, do_sample=True, top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # print(f"LLM call took: {time.time()-start}")
        return self.tokenizer.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)


def guess_smell_type(review: str):
    proc_review = review.lower()
    if "feature envy" in proc_review:
        return "feature_envy"
    elif "long method" in proc_review:
        return "long method"
    return "unknown"

def detect_code_smell_instances(data):
    """heuristic method to detect reviews that mention code smells."""
    smelly_commits = []
    for i, inst in enumerate(data):
        if "code smell" in inst['msg'].lower():
            inst['orig_idx'] = i
            inst["violation_type"] = "code_smell"
            inst["violation_subtype"] = guess_smell_type(inst['msg'])
            smelly_commits.append(inst)

    return smelly_commits

def detect_priciple_violations(data):
    violating_commits = []
    for i, inst in enumerate(data):
        if "principle" in inst['msg'].lower() and "violated" in inst['msg']:
            inst['orig_idx'] = i
            inst["violation_type"] = "principle_violated"
            inst["violation_subtype"] = None
            violating_commits.append(inst)

    return violating_commits

PYTHON_SMELL_NAMES = {
    "long_methods": "Long Method",
    "long_parameter": "Long Parameter List",
    "long_branches": "Long Branch",
    "many_attributes_in_class": "Class with Too Many Attributes",
    "many_methods_in_class": "Class with Too Many Methods",
    "useless_try_except_clauses": "Useless Try/Except Clauses",
    "shotgun_surgery": "Shotgun Surgery",
    "class_cohesion": "Class Cohesion",
    "code_complexity": "Complex Code",
    "long_lambda": "Long Lambda",
    "long_list_comprehension": "Long List Comprehension"
}
PYTHON_SMELL_ELABORATIONS = {
    "long_methods": "longest method at line {lineno} contaning {metric} lines",
    "long_parameter": "method at line {lineno} containing {metric} parameters",
    "long_branches": "longest branch begining at line {lineno} containing {metric} branches",
    "many_attributes_in_class": "class at line {lineno} having {metric} attributes",
    "many_methods_in_class": "class at line {lineno} having {metric} methods",
    "useless_try_except_clauses": "{num_useless_try_except_clauses} try/except clauses being useless out of {total_try_except_clauses}",
    "shotgun_surgery": "class {class_name} having {num_external_function_calss} external function calls",
    "class_cohesion": "Class Cohesion",
    "code_complexity": "Complex Code",
    "long_lambda": "Long Lambda",
    "long_list_comprehension": "Long List Comprehension"
}
def load_py_smells_file(smells_folder: str, codes_folder: str):
    smell_summaries = {}
    for filename in os.listdir(smells_folder):
        filepath = os.path.join(smells_folder, filename)
        code_smell_data = json.load(open(filepath, "r"))
        project_path = code_smell_data["project_path"]
        codelines = open(os.path.join(codes_folder, project_path, project_path+".py")).readlines()
        smell_summaries[project_path] = []
        for smell_type, smell_data in code_smell_data["pylint_output"].items():
            num_smelly_items = 0
            total_items = 0
            for k, v in smell_data.items():
                if k.startswith("num"): num_smelly_items = v
                elif k.startswith("total"): total_items = v
                elif isinstance(v, dict): smell_details = v
            if num_smelly_items > 0:
                smell_name = PYTHON_SMELL_NAMES[smell_type]
                elaboration = PYTHON_SMELL_ELABORATIONS[smell_type].format(**smell_details)
                smell_summaries[project_path].append(f"Contains {num_smelly_items} {'instance' if num_smelly_items == 1 else 'instances'} of {smell_name} smell with {elaboration}.")
        smell_summaries[project_path] = "\n".join(smell_summaries[project_path])

        print(smell_summaries[project_path]+"\n")

    return smell_summaries

# main
if __name__ == "__main__":
    smell_summaries = load_py_smells_file("/home/arnaik/Pyscent/output/json_logs/", "/home/arnaik/code-review-test-projects/python")
    # train_data = read_jsonl("./data/Comment_Generation/msg-train.jsonl")
    # smelly_commits = detect_code_smell_instances(train_data)
    # with open("./data/Comment_Generation/code-smells-train.json", 'w') as f:
    #     json.dump(smelly_commits, f, indent=4)
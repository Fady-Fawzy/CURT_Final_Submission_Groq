"""Approved inventory functions and Groq tool declarations."""

from data_access import get_by_category, get_part


def check_stock(item_name):
    """Return a part's quantity and storage location."""
    part = get_part(item_name)
    if part is None:
        return {
            "found": False,
            "message": f"The item '{item_name}' was not found in the inventory.",
        }

    return {
        "found": True,
        "item_name": part["name"],
        "quantity": part["quantity"],
        "location": part["location"],
    }


def list_by_category(category):
    """Return all parts in a category."""
    return get_by_category(category)


def flag_shortage(item_name):
    """Print a simple low-stock note for a part."""
    print(f"LOW STOCK FLAG: {item_name}")
    return {"logged": True, "item_name": item_name}


AVAILABLE_TOOLS = {
    "check_stock": check_stock,
    "list_by_category": list_by_category,
    "flag_shortage": flag_shortage,
}


GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Get the quantity and storage location for one inventory item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The inventory part name, such as Brake Pads.",
                    }
                },
                "required": ["item_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_by_category",
            "description": "List the inventory items in one category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The inventory category, such as Electrical.",
                    }
                },
                "required": ["category"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_shortage",
            "description": "Log a low-stock flag when the user asks to report or flag a shortage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The inventory part name with a shortage.",
                    }
                },
                "required": ["item_name"],
                "additionalProperties": False,
            },
        },
    },
]

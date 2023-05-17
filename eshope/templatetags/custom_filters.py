from django import template

register = template.Library()

@register.filter
def get_range(value):
    return range(value)
def get_rows(value, row_size):
    rows = [value[i:i+row_size] for i in range(0, len(value), row_size)]
    return rows

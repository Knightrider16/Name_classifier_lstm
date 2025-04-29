def category_from_output(output, all_categories):
    _, top_i = output.topk(1)
    category_index = top_i.item()
    return all_categories[category_index], category_index

const items = [
  { id: 1, name: "Apple", category: "Fruit" },
  { id: 2, name: "Banana", category: "Fruit" },
  { id: 3, name: "Carrot", category: "Vegetable" },
];

const grouped = items.reduce((acc, item) => {
  if (!acc[item.category]) {
    acc[item.category] = { name: [], count: 0 };
  }
  acc[item.category].name.push(item.name);
  acc[item.category].count++;

  return acc;
}, {});

console.log(grouped);

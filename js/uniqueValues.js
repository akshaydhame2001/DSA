const arr = [
  { name: "Akshay", age: 23 },
  { name: "Pratik", age: 20 },
  { name: "Tejas", age: 23 },
  { name: "Mayur", age: 17 },
  { name: "Abhijit", age: 17 },
  { name: "Sujal", age: 20 },
];

const temp = [];

for (const obj of arr) {
  if (!temp.includes(obj.age)) {
    temp.push(obj.age);
  }
}

console.log(temp);

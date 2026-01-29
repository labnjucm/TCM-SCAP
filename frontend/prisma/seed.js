const { PrismaClient } = require("@prisma/client");
const bcrypt = require("bcrypt");

const prisma = new PrismaClient();

async function main() {
  const email = "admin@example.com";
  const password = "admin123";

  const passwordHash = await bcrypt.hash(password, 10);

  await prisma.user.upsert({
    where: { email },
    update: {
      passwordHash,
      role: "admin",
      canDocking: true,
      canMD: true,
      canOrca: true,
    },
    create: {
      email,
      passwordHash,
      role: "admin",
      canDocking: true,
      canMD: true,
      canOrca: true,
    },
  });

  console.log("✅ Admin ensured:", email);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });

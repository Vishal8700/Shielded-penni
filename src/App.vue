<script setup>
import { RouterLink, RouterView } from 'vue-router'
// import HelloWorld from './components/HelloWorld.vue'
import frontend from './views/Frontend.vue'

</script>

<template>
  <header>
    <div>
      <nav>
        <RouterLink to="/">frontend</RouterLink>

      </nav>
    </div>
  </header>

  <RouterView />
</template>

<style scoped></style>

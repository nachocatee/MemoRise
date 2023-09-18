import { StackNavigationProp } from "@react-navigation/stack";
import React from "react";
import { View, Dimensions, Text } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import { RootStackParamList } from "../../App";
import ConfirmBtn from "../../components/Button/ConfirmBtn";
import MainHeader from "../../components/Header/MainHeader";

type MainNavigationProp = StackNavigationProp<RootStackParamList, "Login">;

type Props = {
  navigation: MainNavigationProp;
};

const MainScreen: React.FC<Props> = ({ navigation }) => {
  const ImsiHandler = () => {
    navigation.navigate("SignUp");
  };

  const screenWidth = Dimensions.get("window").width;

  return (
    <LinearGradient
      // colors={["#000000", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <MainHeader />
      <View>
        <ConfirmBtn onPress={ImsiHandler}>뒤로가기</ConfirmBtn>
        <Text style={{ color: "#000" }}>{screenWidth}</Text>
      </View>
    </LinearGradient>
  );
};

export default MainScreen;

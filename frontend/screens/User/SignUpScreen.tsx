import React from "react";
import {
  View,
  Text,
  StyleSheet,
  Image,
  TextInput,
  ScrollView,
} from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { StackNavigationProp } from "@react-navigation/stack";

import ConfirmBtn from "../../components/Button/ConfirmBtn";
import HighlightHeader from "../../components/Header/HighlightHeader";
import ProfilePic from "../../components/ProfilePic";
import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";
import { RootStackParamList } from "../../App";

type SignUpScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  "SignUp"
>;

type Props = {
  navigation: SignUpScreenNavigationProp;
};

const SignUpScreen: React.FC<Props> = ({ navigation }) => {
  const SignUpHandler = () => {
    navigation.navigate("Main");
  };

  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <HighlightHeader />
      <View style={styles.container}>
        <ScrollView
          contentContainerStyle={{
            flexGrow: 1,
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <View>
            <View style={{ alignItems: "center" }}>
              <View style={styles.profilebox}>
                <ProfilePic />
                <Image
                  source={require("../../assets/icons/album.png")}
                  style={styles.album}
                />
              </View>
            </View>
            <View style={styles.inputBox}>
              <Text style={styles.text}>닉네임</Text>
              <TextInput style={styles.input} />
              <Text style={styles.infoText}>
                한글, 영어, 숫자만 사용할 수 있어요. (최대 10자)
              </Text>
            </View>
          </View>
          <View style={styles.buttonContainer}>
            <ConfirmBtn onPress={SignUpHandler}>회원가입</ConfirmBtn>
          </View>
        </ScrollView>
      </View>
    </LinearGradient>
  );
};

export default SignUpScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  text: {
    fontSize: calculateDynamicWidth(20),
    fontFamily: "Pretendard-SemiBold",
    color: Colors.text,
    letterSpacing: -0.5,
  },
  profilebox: {
    width: calculateDynamicWidth(103),
    marginTop: "20%",
  },
  album: {
    width: calculateDynamicWidth(31),
    height: calculateDynamicWidth(31),
    position: "absolute",
    bottom: 0,
    right: 0,
  },
  inputBox: {
    width: calculateDynamicWidth(285),
    marginTop: 50,
  },
  input: {
    borderBottomWidth: 2,
    borderBottomColor: Colors.text,
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
    marginBottom: 10,
  },
  infoText: {
    fontSize: calculateDynamicWidth(13),
    color: Colors.text,
    opacity: 0.5,
    fontFamily: "Pretendard-Light",
    letterSpacing: -0.5,
  },
  buttonContainer: {
    marginBottom: "10%",
  },
});

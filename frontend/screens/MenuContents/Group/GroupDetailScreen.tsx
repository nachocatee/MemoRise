import axios from "axios";
import React, { useEffect, useState } from "react";
import { View, Text, Image, Pressable, ScrollView } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { RouteProp } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { BACKEND_URL } from "../../../util/http";
import { RootStackParamList } from "../../../App";
import { styles } from "./GroupStyle";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import UserList from "../../../components/UserList";

type GroupDetailScreenProps = {
  route: RouteProp<RootStackParamList, "GroupDetail">;
  navigation: StackNavigationProp<RootStackParamList, "GroupDetail">;
};

type GroupData = {
  name: string;
  me: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  owner: {
    userSeq: number;
    email: string;
    nickname: string;
    profile: string;
  };
  members: [
    {
      userSeq: number;
      email: string;
      nickname: string;
      profile: string;
    }
  ];
  password: string;
};

const GroupDetailScreen: React.FC<GroupDetailScreenProps> = ({
  route,
  navigation,
}) => {
  const { teamSeq, userSeq } = route.params;
  const [groupData, setGroupData] = useState<GroupData | null>(null);

  // 그룹 상세 데이터 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          url: BACKEND_URL + `/teams/${teamSeq}/${userSeq}`,
        });
        setGroupData(res.data);
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, [groupData]);

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        {groupData && !groupData.owner ? (
          // 유저 == 그룹장일 때
          <>
            <View style={styles.titleContainer}>
              <Text style={styles.title}>{groupData.name}</Text>
              <Pressable
                onPress={() => navigation.navigate("GroupSetting")}
                style={styles.settingContainer}
              >
                <Image
                  style={styles.settingIcon}
                  source={require("../../../assets/icons/setting_sm.png")}
                />
                <Text style={styles.settingText}>그룹 설정</Text>
              </Pressable>
            </View>
            <View style={styles.memberContainer}>
              <Pressable
                onPress={() =>
                  navigation.navigate("InviteUser", {
                    teamSeq: teamSeq,
                    teamName: groupData.name,
                  })
                }
                style={[
                  styles.settingContainer,
                  { marginBottom: calculateDynamicWidth(25) },
                ]}
              >
                <View style={styles.memberImageContainer}>
                  <Image
                    style={styles.memberImagebg}
                    source={require("../../../assets/image/addmember.png")}
                  />
                </View>
                <Text style={styles.memberText}>초대하기</Text>
              </Pressable>
            </View>
          </>
        ) : null}
        {groupData && (
          <ScrollView contentContainerStyle={styles.memberWrap}>
            {/* 내 정보 */}
            <UserList
              profileUri={groupData.me.profile}
              nickname={groupData.me.nickname}
            />
            <Pressable>
              <Image
                style={styles.meIcon}
                source={require("../../../assets/image/me.png")}
              />
            </Pressable>
          </ScrollView>
        )}
      </View>
    </LinearGradient>
  );
};

export default GroupDetailScreen;

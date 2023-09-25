import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import axios from "axios";
import { useEffect, useState } from "react";
import { View, Text, Image, Pressable, ScrollView } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { useSelector } from "react-redux";
import SmallBtn from "../../../components/Button/SmallBtn";
import GroupBox from "../../../components/GroupBox";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { RootState } from "../../../store/store";
import { BACKEND_URL } from "../../../util/http";
import { styles } from "./GroupStyle";

type RootStackParamList = {
  FindGroup: undefined;
};

type GroupData = [
  {
    teamSeq: number;
    teamName: string;
    myProfile: string;
    ownerProfile: string;
    memberProfiles: [string];
    owner: boolean;
  }
];

const MyGroupScreen = () => {
  const navigation = useNavigation<StackNavigationProp<RootStackParamList>>();
  const [groupData, setGroupData] = useState<GroupData | null>(null);
  // const userId = useSelector((state: RootState) => state.userInfo.id);

  // 내 그룹 목록 가져오기
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios({
          method: "GET",
          // url: BACKEND_URL + `/user/${userId}/my-teams`,
          url: BACKEND_URL + `/user/26/my-teams`, // 더미 데이터
        });
        setGroupData(res.data);
        console.log("조회 성공");
      } catch (err) {
        console.log(err);
      }
    };
    fetchData();
  }, []);

  console.log(groupData);

  //  그룹 편집 버튼
  const [isEditGroup, setEditGroup] = useState(false);

  const EditGroupsHandler = () => {};

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>내 그룹</Text>
          {groupData && groupData[0] && (
            <SmallBtn onPress={EditGroupsHandler}>편집</SmallBtn>
          )}
        </View>
        {groupData && groupData[0] ? (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.groupContainer}
          >
            {groupData.map((group, idx) => (
              <View key={idx}>
                <GroupBox teamName={group.teamName} />
              </View>
            ))}
          </ScrollView>
        ) : (
          <View style={styles.emptyContainer}>
            <Image
              style={styles.group}
              source={require("../../../assets/icons/group.png")}
            />
            <Text style={styles.empty}>참여한 그룹이 없습니다</Text>
            <Pressable
              style={({ pressed }) =>
                pressed
                  ? [styles.findGroupBtn, styles.pressed]
                  : styles.findGroupBtn
              }
              onPress={() => navigation.navigate("FindGroup")}
            >
              <Text style={styles.btnText}>그룹 찾기</Text>
            </Pressable>
          </View>
        )}
      </View>
    </LinearGradient>
  );
};

export default MyGroupScreen;

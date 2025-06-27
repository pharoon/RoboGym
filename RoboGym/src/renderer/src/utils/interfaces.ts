import { SetStateAction } from "react"

interface Basic {
    userProfile:Partial<Profile>

}

export interface TrainProps extends Basic{
}

export interface TestPageProps extends Basic{
}

export interface  HomePageProps extends Basic{
  setUserProfile:React.Dispatch<SetStateAction<Profile | {}>>
}

export interface DeleteModelProps extends Basic {
  showDeleteModal: boolean
  setShowdeleteModal: React.Dispatch<SetStateAction<boolean>>
}

export interface AllDataProps extends Basic {
    
}

export interface AnalyticsProps extends Basic{
}

export type Profile = {
    username:string
    email:string
    user_id:string
}

export type Model = {
    algorithm:string
    created_at:string
    id: number
    model_path: string 
    name: string
    robotic_arm:string
}